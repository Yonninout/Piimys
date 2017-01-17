package projetstl.com.Piimys;
        import android.content.Context;
        import android.content.res.AssetManager;
        import android.graphics.Bitmap;
        import android.graphics.BitmapFactory;
        import android.support.v7.app.AppCompatActivity;
        import android.os.Bundle;
        import android.util.Log;
        import android.view.View;
        import android.widget.Button;
        import android.widget.ImageView;
        import android.widget.Toast;

        import org.bytedeco.javacpp.opencv_core;
        import org.bytedeco.javacpp.opencv_nonfree;

        import static org.bytedeco.javacpp.opencv_features2d.*;
        import static org.bytedeco.javacpp.opencv_highgui.*;
        import org.bytedeco.javacpp.opencv_core.Mat;
        import org.bytedeco.javacpp.opencv_features2d.KeyPoint;

        import java.io.File;
        import java.io.FileOutputStream;
        import java.io.IOException;
        import java.io.InputStream;
        import java.util.Arrays;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {


    // SIFT keypoint features
    private static final int N_FEATURES = 0;
    private static final int N_OCTAVE_LAYERS = 3;
    private static final double CONTRAST_THRESHOLD = 0.04;
    private static final double EDGE_THRESHOLD = 10;
    private static final double SIGMA = 1.6;

    private opencv_nonfree.SIFT SiftDesc;
    private String filePath;

    public static File ToCache(Context context, String Path, String fileName) {
        InputStream input;
        FileOutputStream output;
        byte[] buffer;
        String filePath = context.getCacheDir() + "/" + fileName;
        File file = new File(filePath);
        AssetManager assetManager = context.getAssets();

        try {
            input = assetManager.open(Path);
            buffer = new byte[input.available()];
            input.read(buffer);
            input.close();

            output = new FileOutputStream(filePath);
            output.write(buffer);
            output.close();
            return file;

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String refFile = "Pepsi_10.jpg";
        this.filePath = this.ToCache(this, "images" + "/" + refFile, refFile).getPath();


        ImageView imageView = (ImageView) findViewById(R.id.Pictures_ImageView);
        Bitmap bitmap = BitmapFactory.decodeFile(filePath);
        imageView.setImageBitmap(bitmap);

        Button keypointsButton = (Button) findViewById(R.id.Analyse_button);

        keypointsButton.setOnClickListener(this);

    }

    @Override
    public void onClick(View view) {

        //init sift
        SiftDesc = new opencv_nonfree.SIFT(N_FEATURES, N_OCTAVE_LAYERS, CONTRAST_THRESHOLD, EDGE_THRESHOLD, SIGMA);

        BFMatcher matcher = new BFMatcher();

        String imageNames[] = new String[0];
        try {
            imageNames = this.getAssets().list("TestImage");
        } catch (IOException e) {
            e.printStackTrace();
        }


        String filePaths[] = new String[imageNames.length];
        Mat images[] = new Mat[imageNames.length];
        KeyPoint keypoints[] = new KeyPoint[imageNames.length];
        for (int i = 0; i<imageNames.length;i++){
            //init all keypoints
            keypoints[i] = new KeyPoint();
        }

        DMatchVectorVector matches[] = new DMatchVectorVector[imageNames.length];
        Mat[] descriptors = new Mat[imageNames.length];
        DMatchVectorVector bestMatches[] = new DMatchVectorVector[imageNames.length];


        Mat imgRef = imread(this.ToCache(this,"images/Pepsi_10.jpg", "Pepsi_10.jpg").getPath());
        KeyPoint keyPointRef = new KeyPoint();
        Mat descriptorsRef = new Mat();

        SiftDesc = new opencv_nonfree.SIFT(N_FEATURES, N_OCTAVE_LAYERS, CONTRAST_THRESHOLD, EDGE_THRESHOLD, SIGMA);

        // init detection for image to find
        SiftDesc.detect(imgRef, keyPointRef);
        SiftDesc.compute(imgRef,keyPointRef, descriptorsRef);

        for (int i = 0; i < imageNames.length; i++) {
            //calcul of all matches from all images
            matches[i] = new DMatchVectorVector();
            descriptors[i] = new opencv_core.Mat();
            String refFile = imageNames[i];
            filePaths[i] = this.ToCache(this, "TestImage" + "/" + refFile, refFile).getPath();
            images[i] = imread(filePaths[i]);

            //Work on all Images
            SiftDesc.detect(images[i],keypoints[i]);
            SiftDesc.compute(images[i],keypoints[i],descriptors[i]);
            matcher.knnMatch(descriptorsRef,descriptors[i],matches[i],2);

            //Refine all Matches
            bestMatches[i] = refineMatches(matches[i]);
        }

        Float distances[] = new Float[bestMatches.length];

        for (int j = 0; j<bestMatches.length; j++){
            distances[j] = 0f;
            for (int i = 0; i < bestMatches[j].size(); i++){
                distances[j] += bestMatches[j].get(i,0).distance();
            }
            //distances[j] /= bestMatches[j].size();
        }
        String matchBest =  imageNames[findMin(distances)];
        Log.d("info", "onClick: " + Arrays.toString(distances));
        Toast.makeText(this, "Best matches is : "+matchBest,Toast.LENGTH_SHORT).show();
    }

    private int findMin (Float tab[]){
        int index = 0;
        for (int i = 0; i < tab.length; i ++){
            if ((tab[index] > tab[i]) && (tab[i] != 0f)){
                index = i;
            }
            System.out.println(index);
        }
        return index;
    }

    private static DMatchVectorVector refineMatches(DMatchVectorVector oldMatches) {
        // Ratio of Distances
        double RoD = 0.6;
        DMatchVectorVector newMatches = new DMatchVectorVector();

        // Refine results 1: Accept only those matches, where best dist is < RoD
        // of 2nd best match.
        int sz = 0;
        newMatches.resize(oldMatches.size());

        double maxDist = 0.0, minDist = 1e100; // infinity

        for (int i = 0; i < oldMatches.size(); i++) {
            newMatches.resize(i, 1);
            if (oldMatches.get(i, 0).distance() < RoD
                    * oldMatches.get(i, 1).distance()) {
                newMatches.put(sz, 0, oldMatches.get(i, 0));
                sz++;
                double distance = oldMatches.get(i, 0).distance();
                if (distance < minDist)
                    minDist = distance;
                if (distance > maxDist)
                    maxDist = distance;
            }
        }
        newMatches.resize(sz);

        // Refine results 2: accept only those matches which distance is no more
        // than 3x greater than best match
        sz = 0;
        DMatchVectorVector brandNewMatches = new DMatchVectorVector();
        brandNewMatches.resize(newMatches.size());
        for (int i = 0; i < newMatches.size(); i++) {
            // Since minDist may be equal to 0.0, add some non-zero value
            if (newMatches.get(i, 0).distance() <= 3 * minDist) {
                brandNewMatches.resize(sz, 1);
                brandNewMatches.put(sz, 0, newMatches.get(i, 0));
                sz++;
            }
        }
        brandNewMatches.resize(sz);
        return brandNewMatches;
    }
}
package mnist;

import org.jblas.DoubleMatrix;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MnistLoader {

    /** the following constants are defined as per the values described at http://yann.lecun.com/exdb/mnist/ **/

    private static final int OFFSET_SIZE = 4; //in bytes

    private static final int LABEL_MAGIC = 2049;
    private static final int IMAGE_MAGIC = 2051;

    private static final int NUMBER_ITEMS_OFFSET = 4;
    private static final int ITEMS_SIZE = 4;

    private static final int NUMBER_OF_ROWS_OFFSET = 8;
    private static final int ROWS_SIZE = 4;
    public static final int ROWS = 28;

    private static final int NUMBER_OF_COLUMNS_OFFSET = 12;
    private static final int COLUMNS_SIZE = 4;
    public static final int COLUMNS = 28;

    private static final int IMAGE_OFFSET = 16;
    private static final int IMAGE_SIZE = ROWS * COLUMNS;

    /**
     * Gets the expected neural network output for the specified digit.
     *
     * @param digit Digit for which we want the expected output.
     * @return Array containing the expected output for the specified digit.
     */
    private double[] getOutputFor(int digit) {
        double[] output = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        output[digit] = 1;
        return output;
    }

    /**
     * Converts the bytes from the file to a usable array of image data.
     *
     * @param bytes Byte array that was loaded from the file.
     * @return Array of doubles representing the processed image.
     */
    private double[] getInputFor(byte[] bytes) {
        double[] imageData = new double[bytes.length];
        for(int i = 0; i < imageData.length; i++) {
            imageData[i] = bytes[i] & 0xFF; //convert to unsigned
        }
        imageData = otsu(imageData);
        return imageData;
    }

    /**
     * Run the Otsu Threshold algorithm to convert gray scale image to black and white.
     * NOTE: This can be changes to eliminate noise in other ways.
     */
    private static double[] otsu(double[] input) {
        int[] histogram = new int[256];

        for(double datum : input) {
            histogram[(int) datum]++;
        }

        double sum = 0;
        for(int i = 0; i < histogram.length; i++) {
            sum += i * histogram[i];
        }

        double sumB = 0;
        int wB = 0;
        int wF;

        double maxVariance = 0;
        int threshold = 0;

        int i = 0;
        boolean found = false;

        while(i < histogram.length && !found) {
            wB += histogram[i];

            if(wB != 0) {
                wF = input.length - wB;

                if(wF != 0) {
                    sumB += (i * histogram[i]);

                    double mB = sumB / wB;
                    double mF = (sum - sumB) / wF;

                    double varianceBetween = wB * Math.pow((mB - mF), 2);

                    if(varianceBetween > maxVariance) {
                        maxVariance = varianceBetween;
                        threshold = i;
                    }
                }

                else {
                    found = true;
                }
            }

            i++;
        }

        for(i = 0; i < input.length; i++) {
            input[i] = input[i] <= threshold ? 0 : 1;
        }

        return input;
    }


    /**
     * Loads the MNIST data from the specified file and returns the data in a List form.
     *
     * @return List of Image Data objects that were loaded from the specified MNIST files.
     * @throws IOException
     */
    public List<MnistSet> create(String labelFileName, String imageFileName) throws IOException {
        List<MnistSet> pairs = new ArrayList<>();

        ByteArrayOutputStream labelBuffer = new ByteArrayOutputStream();
        ByteArrayOutputStream imageBuffer = new ByteArrayOutputStream();

        InputStream labelInputStream = new FileInputStream(labelFileName);
        InputStream imageInputStream = new FileInputStream(imageFileName);

        int read;
        byte[] buffer = new byte[16384];

        while((read = labelInputStream.read(buffer, 0, buffer.length)) != -1) {
            labelBuffer.write(buffer, 0, read);
        }

        labelBuffer.flush();

        while((read = imageInputStream.read(buffer, 0, buffer.length)) != -1) {
            imageBuffer.write(buffer, 0, read);
        }

        imageBuffer.flush();

        byte[] labelBytes = labelBuffer.toByteArray();
        byte[] imageBytes = imageBuffer.toByteArray();

        byte[] labelMagic = Arrays.copyOfRange(labelBytes, 0, OFFSET_SIZE);
        byte[] imageMagic = Arrays.copyOfRange(imageBytes, 0, OFFSET_SIZE);

        if(ByteBuffer.wrap(labelMagic).getInt() != LABEL_MAGIC)  {
            labelInputStream.close();
            imageInputStream.close();
            throw new IOException("Bad magic number in label file!");
        }

        if(ByteBuffer.wrap(imageMagic).getInt() != IMAGE_MAGIC) {
            labelInputStream.close();
            imageInputStream.close();
            throw new IOException("Bad magic number in image file!");
        }

        int numberOfLabels = ByteBuffer.wrap(Arrays.copyOfRange(labelBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();
        int numberOfImages = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();

        if(numberOfImages != numberOfLabels) {
            labelInputStream.close();
            imageInputStream.close();
            throw new IOException("The number of labels and images do not match!");
        }

        int numRows = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_ROWS_OFFSET, NUMBER_OF_ROWS_OFFSET + ROWS_SIZE)).getInt();
        int numCols = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_COLUMNS_OFFSET, NUMBER_OF_COLUMNS_OFFSET + COLUMNS_SIZE)).getInt();

        if(numRows != ROWS && numCols != COLUMNS) {
            labelInputStream.close();
            imageInputStream.close();
            throw new IOException("Bad image. Rows and columns do not equal " + ROWS + "x" + COLUMNS);
        }

        for(int i = 0; i < numberOfLabels; i++) {
            int label = labelBytes[OFFSET_SIZE + ITEMS_SIZE + i];
            byte[] byteData = Arrays.copyOfRange(imageBytes, (i * IMAGE_SIZE) + IMAGE_OFFSET, (i * IMAGE_SIZE) + IMAGE_OFFSET + IMAGE_SIZE);

            // add the pair to the array input/output arrays
            DoubleMatrix input = new DoubleMatrix(getInputFor(byteData));
            DoubleMatrix expected = new DoubleMatrix(getOutputFor(label));
            pairs.add(new MnistSet(input, expected));
        }

        labelInputStream.close();
        imageInputStream.close();

        //System.out.println("Loaded " + pairs.size() + " images.");

        return pairs;
    }
}

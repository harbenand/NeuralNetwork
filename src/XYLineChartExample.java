import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.io.File;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 * Created by harshanandrews on 12/21/15.
 */

public class XYLineChartExample extends JFrame {

    public XYLineChartExample() {
        super("Backpropagation");

        JPanel chartPanel = createChartPanel();
        add(chartPanel, BorderLayout.CENTER);

        setSize(640, 480);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
    }

    private JPanel createChartPanel() {
        String chartTitle1 = "Cross Entropy Cost Function";
        //String chartTitle2 = "Cross Entropy Cost Function";
        //String chartTitle3 = "Accuracy (Quadratic)";
        //String chartTitle4 = "Accuracy (Cross-Entropy);
        String xAxisLabel1 = "Images Processed";
        String yAxisLabel1 = "Error";
        //String xAxisLabel2 = "Epoch";
        //String yAxisLabel2 = "Accuracy";

        XYDataset dataset = createDataset();

        JFreeChart chart = ChartFactory.createXYLineChart(chartTitle1,
                xAxisLabel1, yAxisLabel1, dataset);


        // saves the chart as an image files
        File imageFile = new File("Accuracy_Entropy_train_temp.png");
        int width = 640;
        int height = 480;

        try {
            ChartUtilities.saveChartAsPNG(imageFile, chart, width, height);
        } catch (IOException ex) {
            System.err.println(ex);
        }

        return new ChartPanel(chart);
    }

    private XYDataset createDataset() {
        XYSeriesCollection dataset = new XYSeriesCollection();
        Main m = new Main();

        double avg[] = m.average();
        int classification[] = m.classified();
        double acc[] = new double[classification.length];

        //XYSeries series1 = new XYSeries("Error");
        XYSeries series2 = new XYSeries("Accuracy");

            /*for(int i = 0; i < avg.length; i++) {
                series1.add(i, avg[i]);
            }*/

        for(int i = 0; i < classification.length; i++) {
            if(i > 0) {
                acc[i] = (classification[i] * 100.0) / i;
                series2.add(i, acc[i]);
            }
        }

        //dataset.addSeries(series1);
        dataset.addSeries(series2);

        return dataset;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new XYLineChartExample().setVisible(false);
            }
        });
    }
}

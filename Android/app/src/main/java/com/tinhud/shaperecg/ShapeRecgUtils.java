package com.tinhud.shaperecg;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.opencv.ml.StatModel;
import org.opencv.ml.TrainData;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ShapeRecgUtils {

    private static ANN_MLP mANN;

    /**
     * 从Bitmap中获取特征
     * @param input 输入的图片， 只能是背景为白色， 图形为黑色的一个几何图形;
     * @return  返回特征向量
     */
    public static float[] getEigen(Bitmap input) {

        Mat mat_src = new Mat(input.getHeight(), input.getWidth(), CvType.CV_8UC3);
        Utils.bitmapToMat(input, mat_src);

        //二值化后取反
        Mat mat_gray = new Mat(input.getWidth(), input.getHeight(), CvType.CV_8UC1);
        Imgproc.cvtColor(mat_src, mat_gray, Imgproc.COLOR_BGR2GRAY, 1);
        Mat mat_bin = new Mat(input.getWidth(), input.getHeight(), CvType.CV_8UC1);
        Imgproc.threshold(mat_gray, mat_bin, 120, 255, Imgproc.THRESH_BINARY);
        Core.bitwise_not(mat_bin, mat_bin);

        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mat_bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        return getEigen(contours.get(0));
    }

    /**
     * 从轮廓中获取特征
     * @param input 输入的轮廓
     * @return  返回特征向量
     */
    public static float[] getEigen(MatOfPoint input) {

        MatOfPoint2f input2f = new MatOfPoint2f(input.toArray());

        //获取轮廓的最小外接矩形
        RotatedRect rRect = Imgproc.minAreaRect(input2f);

        //特征R:矩形度,  轮廓面积与最小外接矩形的面积之比, 值越接近1表示越接近矩形
        float R = (float)Imgproc.contourArea(input) / (float)(rRect.size.height * rRect.size.width);


        //计算轮廓的矩
        Moments m = Imgproc.moments(input);
        //计算轮廓的质心坐标
        double x0 = m.m10 / m.m00;
        double y0 = m.m01 / m.m00;

        float sum = 0, count = 0;
        Point[] P = input.toArray();
        for (Point p : P) {
            sum += Math.sqrt(Math.pow(p.x - x0, 2) + Math.pow(p.y - y0, 2)); //轮廓质心到轮廓每个点的距离之和
            count += 1.0;
        }
        //区域到质心的平均距离
        float ur = sum / count;

        sum = 0; count = 0;
        for (Point p : P) {
            sum += Math.pow(Math.sqrt(Math.pow(p.x - x0, 2) + Math.pow(p.y - y0, 2)) - ur,2); //轮廓质心到轮廓每个点的距离再减去均值的差的平方之和
            count += 1.0;
        }
        //区域质心到轮廓点的均方差
        float pr = sum / count;
        //特征cDegree: 球状度,  值越大约接近圆(一般大于10)
        float cDegree = ur / pr;


        //获取轮廓的最小外接圆
        Point point = new Point();
        float[] r = new float[1];
        Imgproc.minEnclosingCircle(input2f, point, r);

        //特征C 面积与外接圆面积比值
        float featureC = (float)Imgproc.contourArea(input) / (float)(r[0]*r[0]*Math.PI);


        //特征D 面积与最小包络三角形面积比值
        Mat t = new Mat();
        Imgproc.minEnclosingTriangle(input, t);
        float featureD = (float)Imgproc.contourArea(input) / (float)Imgproc.contourArea(t);



        float[] res = new float[]{R, cDegree, featureC, featureD};

        return res;
    }

    private static float[][] onehotTable = {{1.0f,0,0,0,0}, {0, 1.0f,0,0,0}, {0, 0, 1.0f,0,0},{0, 0, 0, 1.0f,0},{0, 0, 0, 0,1.0f}};

    //样本名称表
    private static final String[] sampleNameTable = {"circle", "diamond", "rectangle", "star", "triangle"};
    //中文名称表
    public static final String[] NameTable = {"圆", "菱形", "矩形", "星形", "三角形"};

    /**
     * 初始化以及训练神经网络
     * @param assetManager asset资源管理器
     */
    public static void train(AssetManager assetManager) {

        mANN = ANN_MLP.create();
        //设置训练模式
        mANN.setTrainMethod(ANN_MLP.BACKPROP);

        //不修改
        mANN.setBackpropWeightScale(0.1);
        mANN.setBackpropMomentumScale(0.1);

        //设置各层节点， 输入层节点4， 隐藏层节点5， 输出层节点5
        Mat layerSizes = new Mat(1, 3, CvType.CV_32SC1);
        layerSizes.put(0,0, new int[]{4,5,5});
        mANN.setLayerSizes(layerSizes);

        //设置激活函数SIGMOID_SYM，  一定要在设置完层结构后再设置激活函数
        mANN.setActivationFunction(ANN_MLP.SIGMOID_SYM);

        //此处修改样本数(40), 输入数(4), 输出数(5)
        Mat trainInputMat = new Mat(25 ,4, CvType.CV_32FC1);
        Mat trainOutputMat = new Mat(25 ,5, CvType.CV_32FC1);
        Mat testMat = new Mat(15, 4, CvType.CV_32FC1);
        int testCount = 0;      //测试数据计数器
        int trainCount = 0;     //训练数据计数器

        for (int i = 0; i < 40; ++i) {
            try {
                String fileName = "sample/" + sampleNameTable[i / 8] + String.valueOf(i % 8) + ".jpg";      //计算文件名
                Bitmap input = BitmapFactory.decodeStream(assetManager.open(fileName));                     //将文件解码为Bitmap
                if ((i % 8 == 1) || (i % 8 == 3) || (i % 8 == 5)) {     //编号1, 3, 5的样本作为测试数据
                    testMat.put(testCount, 0, getEigen(input));
                    testCount++;
                } else {
                    trainInputMat.put(trainCount, 0, getEigen(input));
                    trainOutputMat.put(trainCount, 0, onehotTable[i / 8]);
                    trainCount++;
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }


        //设置终止条件
        mANN.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 1000, 0.00001));

        //开始训练， ROW_SAMPLE意为一行表示一个几何图形的特征向量
        TrainData td = TrainData.create(trainInputMat, Ml.ROW_SAMPLE, trainOutputMat);
        mANN.train(td);


        //评估模型
        Mat res = new Mat();
        mANN.predict(testMat, res, StatModel.COMPRESSED_INPUT);

        float[][] out = new float[15][5];
        for (int i = 0; i < 15; i++) {
            res.get(i, 0, out[i]);
            Log.i("测试结果：", NameTable[argmax(out[i])]);
        }

    }

    /**
     * 根据特征识别几何图形
     * @param input 输入的特征向量
     */
    public static float[] predict(float[] input) {

        Mat testData = new Mat(1, 4, CvType.CV_32FC1);
        testData.put(0, 0, input);
        Mat res = new Mat();
        mANN.predict(testData, res, StatModel.COMPRESSED_INPUT);

        float[] out = new float[5];
        res.get(0, 0, out);

        return out;
    }

    public static int argmax(float[] input) {

        int maxIndex = 0;

        for (int i = 1; i < input.length; i++) {

            if (input[maxIndex] < input[i]) {
                maxIndex = i;
            }

        }

        return maxIndex;

    }


}

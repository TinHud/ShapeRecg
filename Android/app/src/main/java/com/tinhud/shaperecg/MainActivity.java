package com.tinhud.shaperecg;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Handler;
import android.os.Message;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.lqr.picselect.LQRPhotoSelectUtils;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements View.OnClickListener{

    static {
        OpenCVLoader.initDebug();       //简单加载下openCV  没有错误处理
    }

    private TextView mTv;
    private ImageView mIv;
    private Button mBt_S, mBt_R;

    //训练完成后置true
    private boolean trainFlag = false;
    //选择图片后置true
    private boolean selectFlag = false;

    //存储待识别的图片
    private Bitmap mBitmap;


    private Context mContext;


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // 在Activity中的onActivityResult()方法里与LQRPhotoSelectUtils关联
        mLQR.attachToActivityForResult(requestCode, resultCode, data);
    }

    private LQRPhotoSelectUtils mLQR;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mContext = getApplicationContext();

        //初始化控件
        mIv = (ImageView)findViewById(R.id.imageView);
        mTv = (TextView)findViewById(R.id.textView_result);
        mBt_S = (Button)findViewById(R.id.button_select);
        mBt_S.setOnClickListener(this);
        mBt_R = (Button)findViewById(R.id.button_recg);
        mBt_R.setOnClickListener(this);

        //申请读SD卡权限
        requestPermission();

        //开始训练
        mHandler.sendEmptyMessage(1);

        mLQR = new LQRPhotoSelectUtils(this, new LQRPhotoSelectUtils.PhotoSelectListener() {
            @Override
            public void onFinish(File outputFile, Uri outputUri) {
                // 当拍照或从图库选取图片成功后回调
                mBitmap = BitmapFactory.decodeFile(outputFile.getAbsolutePath());
                mIv.setImageBitmap(mBitmap);
                selectFlag = true;
            }
        }, false);
    }

    //权限申请
    public void requestPermission() {

        ActivityCompat.requestPermissions(this,
                new String[]{
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                }, 1);

    }


    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.button_select:
                mLQR.selectPhoto();
                break;
            case R.id.button_recg:
                if (!trainFlag) {
                    Toast.makeText(mContext, "未训练完成", Toast.LENGTH_SHORT).show();
                } else if (!selectFlag) {
                    Toast.makeText(mContext, "未选择图片", Toast.LENGTH_SHORT).show();
                } else {
                    predict();
                }
                break;
            default:
                break;
        }
    }

    private Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            switch (msg.what) {
                case 1:
                    //开始训练
                    train();
                    break;
                case 2:
                    //训练完成
                    trainFlag = true;
                    break;
                case 3:
                    //得到结果
                    float[] res = (float[])msg.obj;
                    String str;
                    str = ShapeRecgUtils.NameTable[ShapeRecgUtils.argmax(res)];
                    mTv.setText(str);
                default:
                    break;
            }
        }

    };

    //开个线程去训练
    private void train() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                ShapeRecgUtils.train(getAssets());
                mHandler.sendEmptyMessage(2);
            }
        }).start();
    }

    //开个线程去识别
    private void predict() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Message msg = new Message();
                    //获取特征向量
                    float[] e = ShapeRecgUtils.getEigen(mBitmap);
                    //识别
                    float[] res = ShapeRecgUtils.predict(e);
                    //将结果发送给handler
                    msg.what = 3;
                    msg.obj = res;
                    mHandler.sendMessage(msg);
                } catch (Exception ex) {
                    Toast.makeText(mContext, "识别出现错误", Toast.LENGTH_SHORT).show();
                    ex.printStackTrace();
                }
            }
        }).start();
    }

}

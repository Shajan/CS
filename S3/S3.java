/*
 * A light weight S3 client library to get/put files on S3.
 *
 * Amazon S3 is a cloud based file store. It is a paid service.
 *
 * This is a thin wrapper that uses S3 REST APIs, to read and write
 * files. Amazon provides a set of java client libraries as well,
 * they are generic, and comes with a ton of code. I found the amazon
 * provided library to be very large, and wrote this class to understand
 * how the basic request/response actually works.
 *
 * This class has been tested on Android, but can be easily ported to
 * be used from other OS.
 *
 * Dec 25, 2013 : Shajan Dasan (twitter @sdasan, sdasan@gmail.com)
 *
 * Usage:
 *     To upload local file /tmp/bar.txt --> bar.txt
 *     boolean success = S3.upload("/tmp/bar.txt", "bar.txt");
 *
 *     To download bar.txt from S3
 *     InputStream is = S3.download("bar.txt");
 *
 */

package com.sdasan.util.s3;

import android.os.AsyncTask;
import android.util.Base64;

import com.twitter.internal.android.util.IoUtils;

import org.apache.http.Header;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.StatusLine;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.entity.FileEntity;
import org.apache.http.impl.client.DefaultHttpClient;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.SimpleTimeZone;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/*
 * AWS Reference
 *    http://docs.aws.amazon.com/AmazonS3/latest/dev/RESTAuthentication.html
 */
public class S3 {
    // Amazon AWS constants
    private static final String S3_ENCODING = "UTF-8";
    private static final String S3_HASH_ALGORITHM = "HmacSHA1";
    private static final String S3_READ = "GET";
    private static final String S3_WRITE = "PUT";

    /*
     * S3 account, bucket and folder related constants
     * update them or you will get an access denied error from Amazon
     *
     * Note that embedding your account id and key in code is not reccomended.
     */
    private static final String S3_BUCKET = "myBucketName";                     // <-- Change to your bucket Name
    private static final String S3_ID = "Your account ID goes here";            // <-- Get an account ID from Amazon
    private static final String S3_KEY = "Key for your S3 Account goes here";   // <-- Add your account's key
    private static final String S3_HOST = S3_BUCKET + ".s3.amazonaws.com";
    private static final String S3_FOLDER = "/MyFolder/";                       // <-- Change to any folder name

    private static SimpleDateFormat sRfc822DateFormat;
    private static boolean sDebug = true;
    private static boolean sCharles = false;

    static {
        sRfc822DateFormat = new SimpleDateFormat("EEE, dd MMM yyyy HH:mm:ss z", Locale.US);
        sRfc822DateFormat.setTimeZone(new SimpleTimeZone(0, "GMT"));
    }

    public static InputStream download(String s3FileName) {
        InputStream in = null;
        try {
            final String date = sRfc822DateFormat.format(new Date());
            final String hash = getReadHash(S3_BUCKET, s3FileName, date);
            final String url = "https://" + S3_HOST + s3FileName;

            final DefaultHttpClient httpClient = new DefaultHttpClient();
            final HttpGet httpGet = new HttpGet(url);
            httpGet.addHeader("Host", S3_HOST);
            httpGet.addHeader("Date", date);
            httpGet.addHeader("Authorization", "AWS " + S3_ID + ":" + hash);

            if (sDebug) {
                final Header[] headers = httpGet.getAllHeaders();
                for (Header header : headers) {
                    android.util.Log.d("S3", header.getName() + " : "
                        + header.getValue());
                }
            }

            final HttpResponse response = httpClient.execute(httpGet);
            final StatusLine statusLine = response.getStatusLine();
            if (sDebug) {
                android.util.Log.d("S3", "statusLine : " + statusLine.toString());
            }

            final HttpEntity entity = response.getEntity();
            in = entity.getContent();

            if (sDebug) {
                final InputStreamReader is = new InputStreamReader(in);
                final BufferedReader br = new BufferedReader(is);
                try {
                    String line = br.readLine();
                    while (line != null) {
                        android.util.Log.d("S3", line);
                        line = br.readLine();
                    }
                } finally {
                    br.close();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return in;
    }

    public static boolean upload(String localFileName, String s3FileName) {
        final S3 s3 = new S3();
        return s3.doUploadTask(localFileName, s3FileName);
    }

    private boolean doUploadTask(String localFileName, String s3FileName) {
        final UploadTask ut = new UploadTask();
        ut.execute(localFileName, s3FileName);
        return true;
    }

    private class UploadTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            final String localFileName = params[0];
            final String s3FileName = params[1];
            try {
                upload(S3_FOLDER + s3FileName, new File(localFileName));
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }
    }

    class MyFileEntity extends FileEntity {
        private File mFile;
        public MyFileEntity(File file, String mimeType) {
            super(file, mimeType);
            mFile = file;
        }
        @Override
        public long getContentLength() {
            return mFile.length();
        }
    }

    private void upload(String localFileName, File s3FileName)
            throws IOException {
        final String date = sRfc822DateFormat.format(new Date());
        final String hash = getWriteHash(S3_BUCKET, s3FileName, date);
        final String urlStr = "http://" + S3_HOST + s3FileName;
        final String mimeType = mimeType(s3FileName);

        if (sDebug) {
            android.util.Log.d("S3", "url " + urlStr);
        }

        HttpURLConnection connection = null;
        BufferedOutputStream out = null;
        InputStream in = null;

        try {
            final URL url = new URL(urlStr);
            if (sCharles) {
                final Proxy proxy = new Proxy(Proxy.Type.HTTP,
                    new InetSocketAddress("172.17.120.121", 8888));
                connection = (HttpURLConnection) url.openConnection(proxy);
            } else {
                connection = (HttpURLConnection) url.openConnection();
            }
            connection.setRequestMethod("PUT");
            connection.setRequestProperty("Content-Type", mimeType);
            connection.setRequestProperty("Host", S3_HOST);
            connection.setRequestProperty("Date", date);
            connection.setRequestProperty("Authorization", "AWS " + S3_ID + ":" + hash);
            // should this be file conent length instead?
            connection.setRequestProperty("Content-Length", Long.toString(localFileName.length()));
            connection.setUseCaches(false);
            connection.setDoOutput(true);

            out = new BufferedOutputStream(connection.getOutputStream());
            in = new FileInputStream(localFileName);
            final byte[] buffer = new byte[1024];
            int nBytes = in.read(buffer);
            while (nBytes != -1) {
                out.write(buffer, 0, nBytes);
                nBytes = in.read(buffer);
            }
            out.flush();

            if (sDebug) {
                final InputStream resp = new BufferedInputStream(connection.getInputStream());
                final BufferedReader respReader = new BufferedReader(new InputStreamReader(resp));
                String line = "";
                while ((line = respReader.readLine()) != null) {
                    android.util.Log.d("S3", line);
                }
                respReader.close();
                final int result = connection.getResponseCode();
                android.util.Log.d("S3", "HTTP Response Code " + result);
            }
        } catch (Exception ignore) {
            ignore.printStackTrace();
        } finally {
            IoUtils.closeSilently(in);
            IoUtils.closeSilently(out);
            connection.disconnect();
        }
    }

    /*
     * Example: getReadHash("myBucket", "/foo/image.jpg");
     *    Base64(HMAC-SHA1(KEY, UTF-8-Encoding-Of(
     *        "GET\n
     *         \n
     *         \n
     *         Tue, 27 Mar 2007 21:15:45 +0000\n
     *         /myBucket/foo/image.jpg")
     */
    private static String getReadHash(String bucket, String s3FileName, String date) {
        final StringBuilder sb = new StringBuilder();
        sb.append(S3_READ);
        sb.append("\n\n\n");
        sb.append(date);
        sb.append("\n/");
        sb.append(bucket);
        sb.append(s3FileName);
        return hashAndBase64Encode(S3_KEY, sb.toString(), S3_HASH_ALGORITHM);
    }

    /*
     * Example: getWriteHash("myBucket", "/foo/image.jpg");
     *    Base64(HMAC-SHA1(KEY, UTF-8-Encoding-Of(
     *        "PUT\n
     *         \n
     *         image/jpeg\n
     *         Tue, 27 Mar 2007 21:15:45 +0000\n
     *         /myBucket/foo/image.jpg")
     */
    private static String getWriteHash(String bucket, String s3FileName, String date) {
        final StringBuilder sb = new StringBuilder();
        sb.append(S3_WRITE);
        sb.append("\n\n");
        sb.append(mimeType(s3FileName));
        sb.append("\n");
        sb.append(date);
        sb.append("\n/");
        sb.append(bucket);
        sb.append(s3FileName);
        return hashAndBase64Encode(S3_KEY, sb.toString(), S3_HASH_ALGORITHM);
    }

    /*
     *  -------------------------------------------------
     *  : VideoType     Extension  MimeType             :
     *  -------------------------------------------------
     *  : Flash          .flv      video/x-flv          :
     *  : MPEG-4         .mp4      video/mp4            :
     *  : iPhone Index   .m3u8     application/x-mpegURL:
     *  : iPhone Segment .ts       video/MP2T           :
     *  : 3GP Mobile     .3gp      video/3gpp           :
     *  : QuickTime      .mov      video/quicktime      :
     *  : A/V Interleave .avi      video/x-msvideo      :
     *  : Windows Media  .wmv      video/x-ms-wmv       :
     *  -------------------------------------------------
     */
    private static String mimeType(String s3FileName) {
        String mimeType = "video/mp4";
        final int i = s3FileName.lastIndexOf('.');
        if (i > 0) {
            final String extension = s3FileName.substring(i + 1);
            switch (extension) {
            case "flv" : mimeType = "video/x-flv"; break;
            case "mp4" : mimeType = "video/mp4"; break;
            case "m3u8" : mimeType = "application/x-mpegURL"; break;
            case "ts" : mimeType = "video/MP2T"; break;
            case "mov" : mimeType = "video/quicktime"; break;
            case "avi" : mimeType = "video/x-msvideo"; break;
            case "wmv" : mimeType = "video/x-ms-wmv"; break;
            default : break;
            }
        }
        return mimeType;
    }

    private static String hashAndBase64Encode(String key, String data, String algorithm) {
        if (sDebug) {
            android.util.Log.d("S3", "key : " + key);
            android.util.Log.d("S3", "data : " + data);
            android.util.Log.d("S3", "algorithm : " + algorithm);
        }
        try {
            return Base64.encodeToString(
                hash(key.getBytes(S3_ENCODING), data.getBytes(S3_ENCODING), algorithm),
                Base64.DEFAULT);
        } catch (Exception ignore) {
            ignore.printStackTrace();
        }
        return null;
    }

    private static byte[] hash(byte[] key, byte[] data, String algorithm) {
        try {
            final SecretKeySpec signingKey = new SecretKeySpec(key, algorithm);
            final Mac mac = Mac.getInstance(algorithm);
            mac.init(signingKey);
            return mac.doFinal(data);
        } catch (Exception ignore) {
            ignore.printStackTrace();
        }
        return null;
    }

/* @TEST
    private static void unitTests() {
        android.util.Log.d("S3", "Test START------------");
        if (!hashAndBase64Encode(
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "GET\n\n\nTue, 27 Mar 2007 19:36:42 +0000\n/johnsmith/photos/puppy.jpg",
                "HmacSHA1").equals("bWq2s1WEIj+Ydj0vQ697zp+IXMU=")) {
            android.util.Log.d("S3", "Error : Hashing is broken");
        }
        if (!mimeType("/foo/video.mp4").equals("video/mp4")) {
            android.util.Log.d("S3", "Error : MimeType is broken");
        }
        android.util.Log.d("S3", "Test END -------------");
    }
*/
}

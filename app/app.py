import os, sys
import numpy as np
import random
import json

from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename # ファイル名をチェックする関数
from PIL import Image
import tensorflow as tf

from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.python.keras.backend import set_session

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Flaskオブジェクトの生成
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ファイル容量制限 : 1MB
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# .があるかどうかのチェックと、拡張子の確認をする関数
# OKなら１、だめなら0
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# '/' へのアクセス
@app.route('/')
def index():
    return redirect(url_for('predict'))

# '/predict' へのアクセス
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # POST
    if request.method == 'POST':

        # ファイルが読み込まれていない場合は'/predict'に戻る
        if 'file' not in request.files:
            flash('No file.')
            return redirect(url_for('predict'))

        # ファイルが読み込まれている場合はそのファイルを読み込む
        file = request.files['file']
        if file.filename == '':
            flash('No file.')
            return redirect(url_for('predict'))

        # 読み込んだファイルを処理する
        if file and is_allowed_file(file.filename):

            # 安全なファイル名を作成して画像ファイルを保存
            filename = secure_filename(file.filename)
            #filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filepath = filename
            file.save(filepath)

            # 学習済みのMobileNetをロード
            # 構造とともに学習済みの重みも読み込まれる
            model = MobileNet(weights='imagenet')
            # model.summary()

            # 引数で指定した画像ファイルを読み込む
            # サイズはVGG16のデフォルトである224x224にリサイズされる
            img = image.load_img(filepath, target_size=(224, 224))

            # 画像ファイルをサーバから削除
            os.remove(filepath)

            # 読み込んだPIL形式の画像をarrayに変換
            x = image.img_to_array(img)

            # 3次元テンソル（rows, cols, channels) を
            # 4次元テンソル (samples, rows, cols, channels) に変換
            # 入力画像は1枚なのでsamples=1でよい
            x = np.expand_dims(x, axis=0)

            # Top-5のクラスを予測する
            # VGG16の1000クラスはdecode_predictions()で文字列に変換される
            pred = model.predict(preprocess_input(x))
            top = decode_predictions(pred, top=5)[0]
            scores = pred[0]

            # 予測結果をリストに格納する
            results = []
            for i in range(5):
                score_rounddown = int(top[i][2]*1000000) / 10000.0
                results.append([top[i][1], score_rounddown])

            return render_template('result.html', results=results)

    return render_template('predict.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# デバッグ中にCSSをキャッシュせず瞬時に値を反映する
@app.context_processor
def add_staticfile():
    def staticfile_cp(fname):
        path = os.path.join(app.root_path, 'static', 'css', fname)
        mtime =  str(int(os.stat(path).st_mtime))
        return '/static/css/' + fname + '?v=' + str(mtime)
    return dict(staticfile=staticfile_cp)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run()

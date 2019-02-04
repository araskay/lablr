import flask
import os
from werkzeug import secure_filename
import flickr_getlabel
import fileutils
import shutil

app = flask.Flask(__name__)
UPLOAD_FOLDER = '/Users/aras/temp/webapp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
fileutils.createdir(UPLOAD_FOLDER)
app.config["CACHE_TYPE"] = "null"
'''
@app.route('/',methods=["GET","POST"])
def home():
    return(flask.render_template('index.html'))
'''


@app.route('/')
def upload():
   return(flask.render_template('upload.html'))

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    
    #print(flask.request.files.getlist("file"))
    for upload in flask.request.files.getlist("file"):
        #print('Got file:',upload)
        filename = upload.filename
        destination = "/".join([UPLOAD_FOLDER, filename])
        #print ("Accept incoming file:", filename)
        print ("Saved to:", destination)
        upload.save(destination)
    top3, plotfile = flickr_getlabel.get_lables(destination,uploadfolder=UPLOAD_FOLDER)
    print('Plotfile=',plotfile)
    return(flask.render_template('upload.html', originalimage = filename, plotimage=plotfile,\
        label1= top3[0], label2=top3[1], label3=top3[2] ))
    #return(send_from_directory(UPLOAD_FOLDER, filename))

@app.route('/plot/<filename>')
def send_plot(filename):
    return flask.send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/image/<filename>')
def send_image(filename):
    return flask.send_from_directory(UPLOAD_FOLDER, filename)
    #return filename
'''
@app.route('/show/<filename>')
def uploaded_file(filename):
    filename = 'http://127.0.0.1:5000/uploads/' + filename
    return render_template('done.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
'''

'''
@app.route('/output')
def image_file():
    img = flask.request.args.get('fileupload')
    lscore = predict.predict(img)
    image_file = os.path.join(app.config['UPLOAD_FOLDER'], img)
    #img = flask.request.files['file']
    #lscore = img
    return(flask.render_template('index.html',\
            lscore=lscore,\
            image_file=image_file))
'''

if __name__ == '__main__':
    app.run()

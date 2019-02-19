import flask
import os
from werkzeug import secure_filename
import flickr_getlabel
import fileutils
import shutil

app = flask.Flask(__name__)
UPLOAD_FOLDER = './temp'
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
   return flask.render_template('upload.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    
    #print(flask.request.files.getlist("file"))
    for upload in flask.request.files.getlist("file"):
        #print('Got file:',upload)
        filename = upload.filename
        destination = "/".join([UPLOAD_FOLDER, filename])
        #print ("Accept incoming file:", filename)
        #print ("Saved to:", destination)
        upload.save(destination)
    labels, plotfile_latent, plotfile_pca = flickr_getlabel.get_lables(destination,uploadfolder=UPLOAD_FOLDER)

    return flask.render_template('upload.html', originalimage = filename,
                                                plotimage_latent=plotfile_latent,
                                                plotimage_pca=plotfile_pca,
                                                label1=labels[0], label2=labels[1], label3=labels[2],
                                                label4=labels[3], label5=labels[4]  )
    #return(send_from_directory(UPLOAD_FOLDER, filename))

@app.route('/plot_latent/<filename>')
def send_plot(filename):
    @flask.after_this_request
    def remove_files(response):
        fileutils.removefile(UPLOAD_FOLDER+'/'+filename)
        return response    
    return flask.send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/plot_pca/<filename>')
def send_plot2(filename):
    @flask.after_this_request
    def remove_files(response):
        fileutils.removefile(UPLOAD_FOLDER+'/'+filename)
        return response  
    return flask.send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/image/<filename>')
def send_image(filename):
    '''
    @flask.after_this_request
    def remove_files(response):
        fileutils.removefile(UPLOAD_FOLDER+'/'+filename)
        return response 
    '''
    return flask.send_from_directory(UPLOAD_FOLDER, filename)

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

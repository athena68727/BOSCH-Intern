import smtplib
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import threading
import time
import json
import os
import winsound

class Mail(threading.Thread):
    def __init__(self,ngImagePath):
        threading.Thread.__init__(self)
        self.ngImagePath=ngImagePath

    def run(self):
        # 创建表头
        msg = MIMEMultipart('related')
        msg['From'] = Header("GH_Contamination_Detector", 'utf-8')
        msg['To'] = Header('FA_Team', 'utf-8')
        msg['Subject'] = Header('G/H Contamination Notification', 'utf-8')
        # 创建正文
        content = \
            """
            <p font-weight:bold>Hello iB2_FA03_Team:</p>
            <p>Please find gear housing contamination pic in below:</p>
            <p><img src="cid:image"></p>
            """
        msgAlternative = MIMEMultipart('alternative')
        msg.attach(msgAlternative)
        msgAlternative.attach(MIMEText(content, 'html', 'utf-8'))
        # choose latest report image
        lists = os.listdir(self.ngImagePath)
        lists.sort(key = lambda x:os.path.getmtime((self.ngImagePath + x)))
        file_path = os.path.join(self.ngImagePath, lists[-1])
        fp = open(file_path, 'rb')
        image = MIMEImage(fp.read())
        fp.close()
        image.add_header('Content-ID', '<image>')
        msg.attach(image)
        try:
            sender = 'fixed-term.Yuyang.TANG@cn.bosch.com'
            receivers = ['fixed-term.Yuyang.TANG@cn.bosch.com']#'xiaochuan.guo2@bosch.com',
            smtp = smtplib.SMTP('rb-smtp-int.bosch.com')
            smtp.sendmail(sender, receivers, msg.as_string())
        except smtplib.SMTPException:
            print('email fail !')
        self.stop()

    def stop(self):
        print("Alarm ends.")

class Andon(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.topic='fa03_grease_check'

    def connect_mqtt(self):
        mqttHost='10.177.241.33'
        mqttPort=1883
        clientId='fa03_grease_check'
        self.client=mqtt.Client(clientId)
        self.client.username_pw_set(username='admin', password='public')
        self.client.connect(mqttHost,mqttPort,60)
        self.client.loop_forever()

    def sendMsg(self,msg):
        payload={'msg':"%s" % msg}
        self.client.publish(self.topic,json.dumps(payload,ensure_ascii=False))


    def run(self):
        self.connect_mqtt()

def raise_alarm():
    '''fa03_andon = Andon()
    fa03_andon.start()
    ngImagePath=os.getcwd() + r'/report/'
    #mail=Mail(ngImagePath)
    #mail.start()
    time.sleep(2)

    fa03_andon.sendMsg("err")
    time.sleep(100)
    fa03_andon.sendMsg("ok")'''


    winsound.Beep(3000,3000)

#raise_alarm()


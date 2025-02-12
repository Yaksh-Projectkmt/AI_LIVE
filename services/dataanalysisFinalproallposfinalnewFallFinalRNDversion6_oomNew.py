import random
from paho.mqtt import client as mqtt_client
import ssl
import paho.mqtt.client as mqtt
import csv
import numpy as np
import pickle,time

broker = 'mqtt.oomcardio.com'
port = 8883
client_id = f'{random.randint(1000000000000000, 2000000000000000)}'
username = 'ranchodrai'
password = 'eSyk1b07B0x942R1cA0oc4cu'




x=[]
y=[]
z=[]
Data=[]
def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc,protocol):
        if rc == 0:
            print("Connected MQTT to Accelerometer")
            topic = "oom/ecg/positionXYZ"
            client.subscribe(topic)

        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id,protocol=mqtt_client.MQTTv5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    client.tls_set_context(context)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client



import ast
def ava(lst):
    return sum(lst) / len(lst)

vara = ''
dictpatient={}
def subscribe(client: mqtt_client):

    def on_message(client, userdata, msg):
        try:
            global x,y,z,vara,dictpatient
            Data = ast.literal_eval((msg.payload.decode()))
            patient = Data["patientId"]
            maindata = []


            with open('outfile', 'wb') as fp:
                pickle.dump(Data, fp)
            x.clear()
            y.clear()
            z.clear()
            for i in range(len(Data["data"])):
                x.append(Data["data"][i]["x"])
                y.append(Data["data"][i]["y"])
                z.append(Data["data"][i]["z"])
            avg = ava(x)
            maindata.append(max(x))

            try:
                if dictpatient[patient]=="":
                    dictpatient.update({patient:""})
            except:
                dictpatient.update({patient:""})

            xdiff = np.diff(x)
            ydiff = np.diff(y)
            zdiff = np.diff(z)

            xdiffavg = ava(xdiff)
            ydiffavg = ava(ydiff)
            zdiffavg = ava(zdiff)
    ##        print(dictpatient)
    ##        print("\n",xdiff,ydiff,zdiff,xdiffavg,ydiffavg,zdiffavg,"  ",sum(xdiff),sum(ydiff),sum(zdiff))
           

            topic = "oom/ecg/ecg-position/"+Data["patientId"]

##        if patient == "63c130ab334508f816a6e2ea":
##            print(x,y,z,abs(max(x)-min(x)),abs(max(y)-min(y)),abs(max(z)-min(z)))

            if len([*filter(lambda x: x >= 32767, x)]) > 0 and abs(max(x)-min(x))>30000:
                vara = "RUNNING"
                dictpatient.update({patient:vara})
                client.publish(topic, "RUNNING")
            elif len([*filter(lambda x: x >= 32767, x)]) > 0:
                vara = "RUNNING"
##                print(vara)
                dictpatient.update({patient:vara})
                client.publish(topic, "RUNNING")
            elif len([*filter(lambda x: x > 21000,x)])>=2 and len([*filter(lambda x: x <= 40000, x)])>0:
                vara = "RUNNING"
                dictpatient.update({patient:vara})
##                print(vara)
                client.publish(topic, "RUNNING")

            elif ((abs(max(x)-min(x))>2100 and abs(max(x)-min(x))<9900)) and ((abs(max(y)-min(y))>2100 and abs(max(y)-min(y))<9900)) and ((abs(max(z)-min(z))>2100 and abs(max(z)-min(z))<9900)):
                    vara = "WALKING"       
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic,"WALKING")

            elif any(value >= 15000 or value <= -13000 for value in z):
                    
                if len([*filter(lambda y: y >= 5000, y)])>=2 and len([*filter(lambda y: y <= 16000, y)])>0:
                    vara = "LAYDOWN_RIGHT"
##                    print(vara)
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic, "LAYDOWN_RIGHT")

                elif len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                    vara = "LAYDOWN_LEFT"
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic, "LAYDOWN_LEFT")

                elif ava(z)>9400 and len([*filter(lambda x: x <= 14000, x)]) > 0:
                    vara = "LAYDOWN_DOWNWARD"
                    dictpatient.update({patient:vara})
##                    print(vara)
                    client.publish(topic, "LAYDOWN_DOWNWARD")
                else:
                    mpuData = 1
                    vara = "LAYDOWN"
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic, "LAYDOWN")

##                    print(mpuData)
            elif any(value >= 13000 or value <= -13000 for value in x):
                if len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                    vara = "LAYDOWN_LEFT"
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic, "LAYDOWN_LEFT")
                elif len([*filter(lambda y: y >= 5000, y)])>=2 and len([*filter(lambda y: y <= 16000, y)])>0:
                    vara = "LAYDOWN_RIGHT"
##                    print(vara)
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic, "LAYDOWN_RIGHT")
                elif ava(z)>9400 and len([*filter(lambda x: x <= 14000, x)]) > 0:
                    vara = "LAYDOWN_DOWNWARD"
                    dictpatient.update({patient:vara})
##                    print(vara)
                    client.publish(topic, "LAYDOWN_DOWNWARD")

                else:
                    mpuData = 2
##                    vara = "STAND_SIT"
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic, "STAND_SIT")

##                    print(mpuData)
##            elif any(value >= 13000 for value in y):
##                mpuData = 3
##                print(mpuData)
##            elif any(value <= -13000 for value in y):
##                mpuData = 4
##                print(mpuData)
##            else:
##                mpuData=0
##                print(vara)
            if ava(z)>9400 and len([*filter(lambda x: x <= 14000, x)]) > 0:
                vara = "LAYDOWN_DOWNWARD"
                dictpatient.update({patient:vara})
##                print(vara)
                client.publish(topic, "LAYDOWN_DOWNWARD")
            elif len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                vara = "LAYDOWN_LEFT"
##                print(vara)
                dictpatient.update({patient:vara})
                client.publish(topic, "LAYDOWN_LEFT")

            elif len([*filter(lambda y: y <= 0, y)])>=2 and len([*filter(lambda y: y <= 5000, y)])>1:
                if len([*filter(lambda x: x >= 0, x)])>2 and len([*filter(lambda y: y <= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 and len([*filter(lambda x: x <= 9000, x)])>2:
                        pass
##                    vara = "LAYDOWN2"
##                    print(vara)
##                    dictpatient.update({patient:vara})
##                    client.publish(topic, "LAYDOWN")

                else:
                    if len([*filter(lambda x: x <= 0, x)])>2 and len([*filter(lambda y: y <= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 :

                        vara = "LAYDOWN"
                        dictpatient.update({patient:vara})
                        client.publish(topic, "LAYDOWN")
                    elif len([*filter(lambda x: x <= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 :

                        vara = "LAYDOWN"
##                        print(vara)
                        dictpatient.update({patient:vara})
                        client.publish(topic, "LAYDOWN")
                    elif ((abs(max(x)-min(x))>2100 and abs(max(x)-min(x))<9900)) and ((abs(max(y)-min(y))>2100 and abs(max(y)-min(y))<9900)) and ((abs(max(z)-min(z))>2100 and abs(max(z)-min(z))<9900)):
                            vara = "WALKING3"       
##                            print(vara)
                            dictpatient.update({patient:vara})
                            client.publish(topic,"WALKING")


                    elif len([*filter(lambda x: x >= 32767, x)]) > 0 and abs(max(x)-min(x))>30000:
                        vara = "RUNNING"
                        dictpatient.update({patient:vara})
                        client.publish(topic, "RUNNING")
                    elif len([*filter(lambda x: x >= 32767, x)]) > 0:
                        vara = "RUNNING"
##                        print(vara)
                        dictpatient.update({patient:vara})
                        client.publish(topic, "RUNNING")
                    elif len([*filter(lambda x: x > 21000,x)])>=2 and len([*filter(lambda x: x <= 40000, x)])>0:
                        vara = "RUNNING"
                        dictpatient.update({patient:vara})
##                        print(vara)
                        client.publish(topic, "RUNNING")

                    else:
                        vara = "STAND_SIT1"
##                        print(vara)
                        dictpatient.update({patient:vara})
                        client.publish(topic, "STAND_SIT")

            elif len([*filter(lambda x: x >= 16900, x)])>1 and len([*filter(lambda x: x < 24000, x)])>0:
                if len([*filter(lambda x: x <= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2 :

                    vara = "LAYDOWN4"
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic, "LAYDOWN")


                elif len([*filter(lambda y: y <= 0, y)])>=2 and ava(x)>15500 and len([*filter(lambda z: z <= 0, z)])>3:
                    vara = "WALKING2"
##                    print(vara)
                    dictpatient.update({patient:vara})
                    client.publish(topic,"WALKING")

    ##                elif len([*filter(lambda y: y >= -1000, y)])>=2 and len([*filter(lambda y: y <= 5000, y)])>1 and len([*filter(lambda y: y >= 0, y)])>3:
##                elif len([*filter(lambda y: y <= 0, y)])>=2 and len([*filter(lambda y: y <= 5000, y)])>1:
##                    vara = "STAND_SIT2"
##                    print(vara)
##                    dictpatient.update({patient:vara})
##                    client.publish(topic, "STAND_SIT")
            else:
                if len([*filter(lambda x: x >= 17000, x)])>len([*filter(lambda x: x < 16500, x)]):
                    walking=1
                    idle=0
                else:
                    idle=1
                    walking=0
                if len([*filter(lambda x: x < 16800, maindata)])>0 and idle==1:
                    
                    if len([*filter(lambda y: y >= -1000, y)])>=2 and len([*filter(lambda y: y <= 5000, y)])>0:
                        if len([*filter(lambda x: x >= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= -15000, z)])>2 :
                            

    ##                        if len([*filter(lambda z: z <= -200, z)])>2 and len([*filter(lambda z: z >= -17000, z)])>2 and len([*filter(lambda x: x <= 2500, x)])>2 :
                            vara = "LAYDOWN"
##                            print(vara)
                            dictpatient.update({patient:vara})
    ##                        print(vara)
                            client.publish(topic, "LAYDOWN")

                        elif len([*filter(lambda x: x >= 0, x)])>2 and len([*filter(lambda y: y >= 0, y)])>2 and len([*filter(lambda z: z <= 0, z)])>2:
                            

    ##                        if len([*filter(lambda z: z <= -200, z)])>2 and len([*filter(lambda z: z >= -17000, z)])>2 and len([*filter(lambda x: x <= 2500, x)])>2 :
                            vara = "STAND_SIT"
##                            print(vara)
                            dictpatient.update({patient:vara})
    ##                        print(vara)
                            client.publish(topic, "STAND_SIT")
    ##                        else:
    ##                           vara = "STAND_SIT"
    ##                           print(vara)
    ##    ##                       print(vara)
    ##
    ##                           dictpatient.update({patient:vara})
    ##                           client.publish(topic, "STAND_SIT")
                    
                    elif len([*filter(lambda y: y >= 5000, y)])>=2 and len([*filter(lambda y: y <= 16000, y)])>0:
                        vara = "LAYDOWN_RIGHT"
##                        print(vara)
    ##                    print(vara)
                        dictpatient.update({patient:vara})
                        client.publish(topic, "LAYDOWN_RIGHT")

                    elif len([*filter(lambda y: y <=-7000, y)])>=2 and len([*filter(lambda y: y >=-15900, y)])>0:
                        vara = "LAYDOWN_LEFT"
##                        print(vara)
                        dictpatient.update({patient:vara})
                        client.publish(topic, "LAYDOWN_LEFT")

                    elif len([*filter(lambda z: z <= -7000, z)])>2 and len([*filter(lambda z: z >= -17000, z)])>2 :
                        vara = "LAYDOWN"
##                        print(vara)
    ##                    print(vara)
                        dictpatient.update({patient:vara})
                        client.publish(topic, "LAYDOWN")
            client.on_message = on_message
        except:
                pass

    client.on_message = on_message



##def run():
##    try:
##        client = connect_mqtt()
##        subscribe(client)
##        client.loop_forever()
##    except:
##        print("Data Failure")
##        client = connect_mqtt()
##        subscribe(client)
##        client.loop_forever()

def run():
    while True:
        try:
            try:
                client = connect_mqtt()
                subscribe(client)
                client.loop_forever()
            except:
                print("Data Failure Outside")
                client = connect_mqtt()
                subscribe(client)
                client.loop_forever()
        except Exception as e:
            print("Outside:",e)
if __name__ == '__main__':
    run()

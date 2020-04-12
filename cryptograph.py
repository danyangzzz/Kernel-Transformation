encoding = 'utf-8'

import datetime
import pandas as pd
from Cryptodome.Cipher import AES
from binascii import b2a_hex,a2b_hex
#秘钥,此处需要将字符串转为字节
key = 'abcdefgh'
#加密内容需要长达16位字符，所以进行空格拼接
def pad(text):
    while len(text) % 16 != 0:
        text += ' '
    return text
#加密秘钥需要长达16位字符，所以进行空格拼接
def pad_key(key):
    while len(key) % 16 != 0:
        key += ' '
    return key

key = pad_key(key)
aes = AES.new(key.encode(), AES.MODE_ECB)

data = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo.csv", index_col=0)
l = data.shape[0]
starttime = datetime.datetime.now()
for i in range(0,l):
    text = str(data.iloc[i,0])+' '+str(data.iloc[i,1])+' '+str(data.iloc[i,2])
    encrypted_text = aes.encrypt(pad(text).encode())
    encrypted_text_hex = b2a_hex(encrypted_text)
    data.iloc[i,3] = encrypted_text_hex
endtime = datetime.datetime.now()
print (endtime - starttime)

starttime = datetime.datetime.now()
for i in range(0,l):
    text = data.iloc[i,3]
     de = str(aes.decrypt(a2b_hex(text)), encoding='utf-8',errors="ignore")
endtime = datetime.datetime.now()
print (endtime - starttime)

data.to_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo.csv",)
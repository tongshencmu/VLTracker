ACCESS_TOKEN=$1

mkdir trackingnet
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1-bwBQP949zSDAESdTWSWvUOjPgESSF9U?alt=media -o trackingnet/TRAIN_0.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1Q-3RQyKEN4qe402-iAbFPTgEga1E054v?alt=media -o trackingnet/TRAIN_1.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1vdEkKNbNa-oTOUOFdui__6_6IdvQp0C3?alt=media -o trackingnet/TRAIN_2.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1Nmep9J3nowTRpl4QYt1uT1RPehovZzBF?alt=media -o trackingnet/TRAIN_3.zip

curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1k2115FsEheFCBTuXIoMd5cmKOcNHpVU9?alt=media -o trackingnet/TRAIN_4.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1LqhUdrdj0DxhvAbpUQDr7LykEiExYODP?alt=media -o trackingnet/TRAIN_5.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1cuXXfGUF2e_Ar6ZlI9GudIqeFRFokIac?alt=media -o trackingnet/TRAIN_6.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1sVrJELlXZT6_f-qi8ijRFkeuBLNX-_4K?alt=media -o trackingnet/TRAIN_7.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1d2_OQZffWHKd1y5j92jVn3X3diEFwpOe?alt=media -o trackingnet/TRAIN_8.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1pftCcR0lOzyORo1iyXwYwBgVP09gcupQ?alt=media -o trackingnet/TRAIN_9.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1KAaYDyvqBeer7YlwBkxyzMM1-Las05EV?alt=media -o trackingnet/TRAIN_10.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1bXwQShtdr21CZMV-DIRNHQQiDsxfv81q?alt=media -o trackingnet/TRAIN_11.zip

curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1YwO9o7370zG9gQUaSQzOSkeoyGNz7nrE?alt=media -o trackingnet/TEST.zip
ACCESS_TOKEN=$1

mkdir tnl2k

curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1Mo3JDSmq-VwY6m9CQe5YDChYoKSlxKkw?alt=media \
-o tnl2k/TNL2K_training.zip

curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/18rXme_u9xSaX171HkoERfLZvXkkoLI61?alt=media \
-o tnl2k/TNL2K_test.tar
#rm cache.zip
python generator.py # for all sessions
#python generator.py <pid> # for 1 session
rsync -avzh cache iblviz:/var/www/ibl_website/website/cache
#zip -r cache.zip cache/
#scp -r cache.zip iblviz:~/
#ssh iblviz "mv ~/cache.zip /var/www/ibl_website/website && cd /var/www/ibl_website/website && unzip cache.zip"

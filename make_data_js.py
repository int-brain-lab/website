from generator import make_data_js

#  update script.js line 666 with the new preprocessings
# symlink the figures folder to the static/cache folder:
# unlink /mnt/h0/kb/code_kcenia/website/static/cache
# ln -s /mnt/h0/kb/viz_figures_new /mnt/h0/kb/code_kcenia/website/static/cache
# then at last run the make data js
# Needed to update data.js to use the correct context data. So basically uncomment the top lines in this file https://github.com/int-brain-lab/website/blob/photometry/static/data.js and comment out line 22
# also changed this line https://github.com/int-brain-lab/website/blob/photometry/static/scripts.js#L690 to reflect the new preprocessing names
make_data_js(
    file_data_js='/mnt/h0/kb/code_kcenia/website/static/data.js',
    cache_dir='/mnt/h0/kb/viz_figures_new'
)
#  run the website
# bash run.sh



# source /mnt/h0/kb/code_kcenia/.venv/bin/activate
# cd  /mnt/h0/kb/code_kcenia/website
# python -c "from generator import make_data_js; make_data_js()"
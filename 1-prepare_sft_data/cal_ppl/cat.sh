DATA=$1
cd $DATA
cat $(ls -v P*.jsonl) > $DATA.jsonl
wc -l $DATA.jsonl
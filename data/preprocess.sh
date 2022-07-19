# preprocess raw xml data
echo "------------------starting to process raw data-----------------"
res=$(python preprocess4xml.py --data GE09 --input_dir ./xml)
echo $res

# get input info for downstream task
res=$(echo $res | grep 'trigger' | sed 's/<.* \([0-9]*\) trigger.* \([0-9]*\) entity.* \([0-9]*\) dep.*/\1 \2 \3/g')
triggers=$(echo $res | awk -F ' ' '{print $1}')  # trigger classes
entity=$(echo $res | awk -F ' ' '{print $2}')  # entity classes
dep=$(echo $res | awk -F ' ' '{print $3}')  # dep classes
echo "----------------------------end--------------------------------"


echo "-------------------starting to construct dict------------------"
python data2inputs.py --dest_dir ./preprocessed --data GE09 --seq_len 125
echo "----------------------------end--------------------------------"


echo "-------------------starting to read embedding------------------"
res=$(python read_embedding.py --dest_dir ./preprocessed --data GE09 --entity_classes $entity)
echo $res

res=$(echo $res | grep 'word')
word_num=$(echo $res | awk -F ' ' '{print $3}')  # the number of words
echo "----------------------------end--------------------------------"

echo "-------------------construct attention labels------------------"
python construct_attention.py --dest_dir ./preprocessed --data GE09 --seq_len 125 --num_label $triggers
echo "----------------------------end--------------------------------"

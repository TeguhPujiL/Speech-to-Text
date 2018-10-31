FILE="asset/Dataset/flac/test/20/20/20-20.trans.txt"
OUT=$(awk '{ print $2 }' $FILE)
i=1
for j in $OUT
do
   if (( $i < 10 )); then
     echo 'asset/Dataset/flac/test/20/20/20-20-000'$i'.flac'
     echo 'Dataset: '$j
     python recognize.py asset/Dataset/flac/test/20/20/20-20-000$i.flac
     i=$((i+1))
     echo '================================================'
   else
     echo 'asset/Dataset/flac/test/20/20/20-20-00'$i'.flac'
     echo 'Dataset: '$j
     python recognize.py asset/Dataset/flac/test/20/20/20-20-00$i.flac
     i=$((i+1))
     echo '==============================================='
   fi
done



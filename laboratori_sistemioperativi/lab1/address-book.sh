#!/usr/bin/bash

view(){
column -t -s',' <  address-book-database.csv | head -n 1
column -t -s',' <  address-book-database.csv | tail -n +2 |sort -k4 -d
}

search(){
IFS=$'\n'
header=$(head -n 1 address-book-database.csv)
count=$(head -n 1 address-book-database.csv | tr "," " " | wc -w )
found=$(grep "$string" address-book-database.csv)
if [ -z "$found" ]; then
echo "Not found"
fi
for line in $(tail -n+2 address-book-database.csv | grep "$string" ); do
for(( i = 1 ; i < count ; i++ ));
do
key=$(head -n 1 address-book-database.csv | cut -d"," -f$i)
value=$(echo $line | grep "$string" | cut -d"," -f$i)
echo "${key^}: $value"
done
echo ""
done
}

insert(){
echo -n "Name:"
read name
echo -n "Surname:"
read surname
echo -n "Phone:"
read phone
echo -n "Mail:"
read mail
if grep -q "$mail" address-book-database.csv
then
echo "Already present"
else
echo -n "City:"
read city
echo -n "Address:"
read address
echo "$name,$surname,$phone,$mail,$city,$address" >> address-book-database.csv
echo "Added"
fi
}

delete(){ 
nrow=$(grep -n "$this"  address-book-database.csv | cut -d : -f 1)
sed -i "${nrow}d" address-book-database.csv
if [ -z $nrow ]
then
echo "Cannot find any record"
else
echo "Deleted"
fi
}

error(){
echo "error: command not found. Please choose one of this: view, search, insert or delete."
}

case $1 in
view)
view
;;
search)
string=$2
search
;;
insert)
insert
;;
delete)
this=$2
delete
;;
*)
error
esac


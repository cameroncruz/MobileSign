sudo apt-get update
sudo apt-get install -y nfs-common
sudo mkdir /mnt/efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-afb88505.efs.us-west-2.amazonaws.com:/ /mnt/efs/
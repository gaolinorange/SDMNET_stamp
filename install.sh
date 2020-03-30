
# Create a virtual environment 
virtualenv --no-site-packages -p python3 ~/sdmnet

# Activate the virtual environment
source ~/sdmnet/bin/activate

# Update pip
pip install -U pip

# download pymesh
mkdir package
wget https://github.com/PyMesh/PyMesh/releases/download/v0.2.1/pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl
mv pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl ./package/pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl

# install tensorflow
pip install tensorflow-gpu==1.12.0

# install necessary library
pip install h5py==2.8.0
pip install pickleshare==0.7.5
pip install scipy==1.3.0
pip install scikit-learn==0.21.2
pip install ./package/pymesh2-0.2.1-cp36-cp36m-linux_x86_64.whl
pip install numpy==1.16.0

echo "installing sucessfully"


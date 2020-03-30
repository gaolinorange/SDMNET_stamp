unzip data.zip
cp ./data/chair* ./code/python/chair_reproduce
cp ./data/plane* ./code/python/plane_reproduce

unzip checkpoint.zip
mv ./checkpoint/05060123_6863bin_1-joint_1-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_chair-trcet_1.0 ./code/python/chair_reproduce
mv ./checkpoint/05050238_2556bin_0-joint_0-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_plane-trcet_1.0 ./code/python/plane_reproduce
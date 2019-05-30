The repository contains code of applying picture to picture model using pretrained model pix2pix. To run this you need few libraries that are listed in requirement.txt. The input is CMP Facade Dataset. There are 2 approaches implemented in following code. 

1. The data is converted into required network shape and network output is saved.
2. Patches of equal size are extracted from input image. Network forward pass is applied and output is combined back into single image.

The code can be run via test_task.py file. Download the pretrained weights from following link: http://efrosgans.eecs.berkeley.edu/pix2pix/models-pytorch
Choose facades_label2photo.sh and save it to checkpoints folder as checkpoints/facades_label2photo_pretrained/latest_net_G.pth

The data set can be downloaded from following link: http://cmp.felk.cvut.cz/~tylecr1/facade/
Download and extract the folder and paste it in root directory of this repository.

To run the network use following command

`python test_task.py --dataroot ./base --direction BtoA --model pix2pix --name facades_label2photo_pretrained --type_eval patches`

--dataroot is the path of downloaded images
--direction tells us we will be converting images to facade. 
--type_eval tells us whether we want to do implementation 1 or 2.
For implementation one use `--type_eval without_patches` and for implementation 2 use `--type_eval patches`.


Contact me at: agk_ahmed@hotmail.com

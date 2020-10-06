python test.py --method=StainNet --test_ssim --model_path=checkpoints/stainnet/Cytopathology/stainnet_staingan_best_psnr_layer3_ch32.pth  --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB
python test.py --method=StainGAN --test_ssim --model_path=checkpoints/staingan/Cytopathology/latest_net_G_A.pth  --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB
python test.py --method=reinhard --test_ssim --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB
python test.py --method=reinhard --test_ssim --random_target  --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB
python test.py --method=macenko  --test_ssim --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB
python test.py --method=macenko  --test_ssim --random_target  --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB
python test.py --method=vahadane --test_ssim --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB
python test.py --method=vahadane --test_ssim --random_target  --source_dir=dataset/Cytopathology/test/testA --gt_dir=dataset/Cytopathology/test/testB

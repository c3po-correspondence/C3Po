from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images, load_megascenes_augmented_images

def reverse_ImgNorm(np_array):
    np_array = np_array * 0.5 + 0.5
    np_array *= 255.0
    np_array = np_array.clip(0, 255).astype(np.uint8)
    return np_array

def get_viz(view1, view2, pred1, pred2):
    def gen_plot(view1, view2, pred1, pred2):
        view1_img = view1["img"].permute(0, 2, 3, 1).cpu().numpy()
        view2_img = view2["img"].permute(0, 2, 3, 1).cpu().numpy()

        B, image_size, _,  _ = view1_img.shape
        fig, axes = plt.subplots(B, 6, figsize=(20, B*6))
        titles = ["img1", "pred1", "conf1", "img2", "pred2", "conf2"]
        for b in range(B):
            if B == 1:
                axes[0].imshow(reverse_ImgNorm(view1_img[b]))
                axes[0].set_title(titles[0])

                axes[1].imshow(pred1["pts3d"][b].cpu().numpy())    
                axes[1].set_title(titles[1])          

                conf1 = pred1["conf"][b].cpu().numpy()
                axes[2].imshow(conf1)    
                axes[2].set_title(titles[2])  

                axes[3].imshow(reverse_ImgNorm(view2_img[b]))
                axes[3].set_title(titles[3])
                
                axes[4].imshow(pred2["pts3d_in_other_view"][b].cpu().numpy())    
                axes[4].set_title(titles[4])          

                conf2 = pred2["conf"][b].cpu().numpy()
                axes[5].imshow(conf2)    
                axes[5].set_title(titles[5]) 
            else:
                axes[b, 0].imshow(reverse_ImgNorm(view1_img[b]))
                axes[b, 0].set_title(titles[0])

                axes[b, 1].imshow(pred1["pts3d"][b].cpu().numpy())    
                axes[b, 1].set_title(titles[1])          

                conf1 = pred1["conf"][b].cpu().numpy()
                axes[b, 2].imshow(conf1)    
                axes[b, 2].set_title(titles[2])  

                axes[b, 3].imshow(reverse_ImgNorm(view2_img[b]))
                axes[b, 3].set_title(titles[3])
                
                axes[b, 4].imshow(pred2["pts3d_in_other_view"][b].cpu().numpy())    
                axes[b, 4].set_title(titles[4])          

                conf2 = pred2["conf"][b].cpu().numpy()
                axes[b, 5].imshow(conf2)    
                axes[b, 5].set_title(titles[5]) 
        plt.tight_layout()
        return fig
    viz = gen_plot(view1, view2, pred1, pred2)
    return viz

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model_name = "/share/phoenix/nfs06/S9/kh775/code/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_224_linear"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    # # load_images can take a list of images or a directory
    image_dir = "/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive"
    landmark = "Abbatiale_d'Ottmarsheim"
    img1_path = join(image_dir, landmark, "plans", "File:Ottmarsheim oben.jpg")
    # img2_path = join(image_dir, landmark, "images", "commons/Abbatiale_d'Ottmarsheim/0/pictures/Abbatiale Saint Pierre et Saint Paul.jpg")
    img2_path = join(image_dir, landmark, "images", "commons/Abbatiale_d'Ottmarsheim/0/pictures/Abbatiale d'Ottmarsheim romanisch.jpg")
    npx_path = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/old/data_train/one_plan_2/coords/00000000.npy"
    # images = load_images([img2_path, img1_path], size=224)  # [img1: ['img': [1, 3, 224, 224], 'true_shape', 'idx', 'instance'], img2]
    
    plan_xys, image_xys = np.load(npx_path)
    images = load_megascenes_augmented_images([img1_path, img2_path], size=512, plan_xys=plan_xys, image_xys=image_xys)
    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=False)
    output = inference([(images[0], images[1])], model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    viz = get_viz(view1, view2, pred1, pred2)
    viz.savefig("/share/phoenix/nfs06/S9/kh775/code/dust3r/inference/512dpt/test1.png")
    # viz.savefig("/share/phoenix/nfs06/S9/kh775/code/dust3r/experiments/aachen_cathedral_matches2.png")
    
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # # next we'll use the global_aligner to align the predictions
    # # depending on your task, you may be fine with the raw output and not need it
    # # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # # retrieve useful values from scene:
    # imgs = scene.imgs
    # focals = scene.get_focals()
    # poses = scene.get_im_poses()
    # pts3d = scene.get_pts3d()
    # confidence_masks = scene.get_masks()

    # # visualize reconstruction
    # scene.show()

    # # find 2D-2D matches between the two images
    # from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    # pts2d_list, pts3d_list = [], []
    # for i in range(2):
    #     conf_i = confidence_masks[i].cpu().numpy()
    #     pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
    #     pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    # reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    # print(f'found {num_matches} matches')
    # matches_im1 = pts2d_list[1][reciprocal_in_P2]
    # matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # # visualize a few matches
    # import numpy as np
    # from matplotlib import pyplot as pl
    # n_viz = 10
    # match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    # img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # cmap = pl.get_cmap('jet')
    # for i in range(n_viz):
    #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    # img_out = "/share/phoenix/nfs06/S9/kh775/code/dust3r/experiments/aachen_cathedral.png"
    # pl.savefig(img_out)
    # pl.show(block=True)

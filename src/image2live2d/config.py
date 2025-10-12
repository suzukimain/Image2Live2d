"""
Consolidated imports for config module.
"""

# Standard library
import os
import copy
import pickle
import random
import string

# Third-party libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from skimage import color
from einops import rearrange
import torch
import torch.nn.functional as F
import onnxruntime as rt
import huggingface_hub
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pytoshop
from pytoshop import layers
import psd_tools
from psd_tools.psd import PSD
import requests



# Declare Execution Providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Download and host the model
model_path = huggingface_hub.hf_hub_download(
    "skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)

def rmbg_get_mask(img, s=1024):
    img = (img / 255).astype(np.float32)
    dim = img.shape[2]
    if dim == 4:
        img = img[..., :3]
        dim = 3
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, dim], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw //
              2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

def assign_tile(row, tile_width, tile_height):
    tile_x = row['x_l'] // tile_width
    tile_y = row['y_l'] // tile_height
    return f"tile_{tile_y}_{tile_x}"

def rmbg_fn(img):
    mask = rmbg_get_mask(img)
    img_rgb = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask_u8 = (mask * 255).astype(np.uint8)
    img_rgba = np.concatenate([img_rgb, mask_u8], axis=2)
    mask_vis = mask_u8.repeat(3, axis=2)
    return mask_vis, img_rgba

def get_foreground(img, h_split, v_split, n_cluster, alpha, th_rate):
    df = rgb2df(img)
    image_width = img.shape[1]
    image_height = img.shape[0]
    mask = rmbg_get_mask(img)
    mask = (mask * 255).astype(np.uint8)
    mask = mask.repeat(3, axis=2)

    num_horizontal_splits = h_split
    num_vertical_splits = v_split
    tile_width = image_width // num_horizontal_splits
    tile_height = image_height // num_vertical_splits

    df['tile'] = df.apply(assign_tile, args=(tile_width, tile_height), axis=1)

    cls = MiniBatchKMeans(n_clusters=n_cluster, batch_size=100)
    cls.fit(df[["r","g","b"]])
    df["label"] = cls.labels_

    mask_df = rgb2df(mask)
    mask_df['bg_label'] = (mask_df['r'] > alpha) & (mask_df['g'] > alpha) & (mask_df['b'] > alpha)

    img_df = df.copy()
    img_df["bg_label"] = mask_df["bg_label"]
    img_df["label"] = img_df["label"].astype(str) + "-" + img_df["tile"]
    bg_rate = img_df.groupby("label").sum()["bg_label"]/img_df.groupby("label").count()["bg_label"]
    img_df['bg_cls'] = (img_df['label'].isin(bg_rate[bg_rate > th_rate].index)).astype(int)
    img_df["a"] = 255
    #img_df.loc[img_df['bg_cls'] == 0, ['a']] = 0
    #img_df.loc[img_df['bg_cls'] != 0, ['a']] = 255
    #img = df2rgba(img_df)

    bg_df = img_df[img_df["bg_cls"] == 0]
    fg_df = img_df[img_df["bg_cls"] != 0] 

    return [fg_df, bg_df]


 


def skimage_rgb2lab(rgb):
    return color.rgb2lab(rgb.reshape(1,1,3))


def rgb2df(img):
  h, w, _ = img.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
  r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
  df = pd.DataFrame({
      "x_l": x_l.ravel(),
      "y_l": y_l.ravel(),
      "r": r.ravel(),
      "g": g.ravel(),
      "b": b.ravel(),
  })
  return df

def mask2df(mask):
  h, w = mask.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
  flg = mask.astype(int)
  df = pd.DataFrame({
      "x_l_m": x_l.ravel(),
      "y_l_m": y_l.ravel(),
      "m_flg": flg.ravel(),
  })
  return df


def rgba2df(img):
  h, w, _ = img.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
  r, g, b, a = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
  df = pd.DataFrame({
      "x_l": x_l.ravel(),
      "y_l": y_l.ravel(),
      "r": r.ravel(),
      "g": g.ravel(),
      "b": b.ravel(),
      "a": a.ravel()
  })
  return df

def hsv2df(img):
    x_l, y_l = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    h, s, v = np.transpose(img, (2, 0, 1))
    df = pd.DataFrame({'x_l': x_l.flatten(), 'y_l': y_l.flatten(), 'h': h.flatten(), 's': s.flatten(), 'v': v.flatten()})
    return df

def df2rgba(img_df):
  # ensure alpha exists
  if "a" not in img_df.columns:
    img_df = img_df.copy()
    img_df["a"] = 255
  # build full index range to enforce identical shapes
  xs = np.arange(int(img_df["x_l"].min()), int(img_df["x_l"].max()) + 1)
  ys = np.arange(int(img_df["y_l"].min()), int(img_df["y_l"].max()) + 1)
  def pivot_fill(col):
    piv = img_df.pivot_table(index="x_l", columns="y_l", values=col, aggfunc="first")
    piv = piv.reindex(index=xs, columns=ys)
    return piv.fillna(0).reset_index(drop=True).values
  r_img = pivot_fill("r")
  g_img = pivot_fill("g")
  b_img = pivot_fill("b")
  a_img = pivot_fill("a")
  df_img = np.stack([r_img, g_img, b_img, a_img], 2).astype(np.uint8)
  return df_img

def df2bgra(img_df):
  if "a" not in img_df.columns:
    img_df = img_df.copy()
    img_df["a"] = 255
  xs = np.arange(int(img_df["x_l"].min()), int(img_df["x_l"].max()) + 1)
  ys = np.arange(int(img_df["y_l"].min()), int(img_df["y_l"].max()) + 1)
  def pivot_fill(col):
    piv = img_df.pivot_table(index="x_l", columns="y_l", values=col, aggfunc="first")
    piv = piv.reindex(index=xs, columns=ys)
    return piv.fillna(0).reset_index(drop=True).values
  r_img = pivot_fill("r")
  g_img = pivot_fill("g")
  b_img = pivot_fill("b")
  a_img = pivot_fill("a")
  df_img = np.stack([b_img, g_img, r_img, a_img], 2).astype(np.uint8)
  return df_img

def df2rgb(img_df):
  xs = np.arange(int(img_df["x_l"].min()), int(img_df["x_l"].max()) + 1)
  ys = np.arange(int(img_df["y_l"].min()), int(img_df["y_l"].max()) + 1)
  def pivot_fill(col):
    piv = img_df.pivot_table(index="x_l", columns="y_l", values=col, aggfunc="first")
    piv = piv.reindex(index=xs, columns=ys)
    return piv.fillna(0).reset_index(drop=True).values
  r_img = pivot_fill("r")
  g_img = pivot_fill("g")
  b_img = pivot_fill("b")
  df_img = np.stack([r_img, g_img, b_img], 2).astype(np.uint8)
  return df_img

def pil2cv(image):
  new_image = np.array(image, dtype=np.uint8)
  if new_image.ndim == 2:
      pass
  elif new_image.shape[2] == 3:
      new_image = new_image[:, :, ::-1]
  elif new_image.shape[2] == 4:
      new_image = new_image[:, :, [2, 1, 0, 3]]
  return new_image

def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


 

# use in-file calc_ciede defined below


def get_cls_update_counts(ciede_df, threshold, cls2counts):
    set_list = [frozenset({cls, tgt}) for cls, tgt in ciede_df[ciede_df['ciede2000'] < threshold][['cls_no', 'tgt_no']].to_numpy()]
    merge_set = []
    while set_list:
        set_a = set_list.pop()
        merged = False
        for i, set_b in enumerate(merge_set):
            if set_a & set_b:
                merge_set[i] |= set_a
                merged = True
                break
        if not merged:
            merge_set.append(set_a)
    merge_dict = {}
    for merge in merge_set:
        max_cls = max(merge, key=cls2counts.get)
        for cls in merge:
            if cls != max_cls:
                merge_dict[cls] = max_cls
    return merge_dict


def get_blur_np(img: np.ndarray, labels: np.ndarray, size, blur=True):
    
    if blur:
        img = rearrange(img, 'n c h w -> h w (n c)').astype(np.float32)
        img = cv2.blur(img, (size, size))
        img = rearrange(img, 'h w (n c) -> n c h w', n=1)
    
    cls = np.unique(labels).reshape(-1, 1, 1, 1)
    masks = np.bitwise_and(img[:, [3]] > 127, cls == labels)
    
    cls_counts = masks.sum(axis=(2, 3), keepdims=True) + 1e-10
    rgb_means = (img[:, :3] * masks).sum(axis=(2, 3), keepdims=True) / cls_counts

    rgb_means = rgb_means.squeeze().tolist()
    cls_list = cls.squeeze().tolist()
    cls_counts = cls_counts.squeeze().tolist()
    
    return rgb_means, cls_list, cls_counts, masks


def get_base_np(img: np.ndarray, loop, cls_num, threshold, size, debug=False, kmeans_samples=-1, device='cpu'):
  rgb_flatten = cluster_samples = img[..., :3].reshape((-1, 3))
  im_h, im_w = img.shape[:2]

  alpha_mask = np.where(img[..., 3] > 127)
  resampled = False
  if rgb_flatten.shape[0] > len(alpha_mask[0]):
    cluster_samples = img[..., :3][alpha_mask].reshape((-1, 3))
    resampled = True

  if len(rgb_flatten) > kmeans_samples and kmeans_samples > 0:
    cluster_samples = shuffle(cluster_samples, random_state=0, n_samples=kmeans_samples)
    resampled = True

  kmeans = MiniBatchKMeans(n_clusters=cls_num).fit(cluster_samples)
  if resampled:
    labels = kmeans.predict(rgb_flatten)
  else:
    labels = kmeans.labels_

  img_np = rearrange([img], 'n h w c -> n c h w').astype(np.float32)
  labels_np = labels.reshape((1, 1, im_h, im_w)).astype(np.float32)

  assert loop > 0
  img_np_ori = np.copy(img_np)
  for i in range(loop):
    rgb_means, cls_list, cls_counts, masks = get_blur_np(img_np, labels_np, size)
    ciede_df = calc_ciede(rgb_means, cls_list)
    cls2rgb, cls2counts, cls2masks = {}, {}, {}
    for c, rgb, count, mask in zip(cls_list, rgb_means, cls_counts, masks):
      cls2rgb[c] = rgb
      cls2counts[c] = count
      cls2masks[c] = mask[None, ...]

    merge_dict = get_cls_update_counts(ciede_df, threshold, cls2counts)
    tgt2merge, notmerged = {}, set(cls_list)
    for k, v in merge_dict.items():
      if v not in tgt2merge:
        tgt2merge[v] = []
        notmerged.remove(v)
      tgt2merge[v].append(k)
      notmerged.remove(k)
    for k in notmerged:
      tgt2merge[k] = []

    for tgtc, srcc_list in tgt2merge.items():
      mask = cls2masks[tgtc]
      for srcc in srcc_list:
        mask = np.bitwise_or(mask, cls2masks[srcc])
      labels_np[mask] = tgtc
      if i != loop - 1:
        for jj in range(3):
          img_np[:, jj][mask[0]] = cls2rgb[tgtc][jj]

  cls_list_final = np.unique(labels_np)
  img_np = img_np_ori
  rgb_means, cls_list, cls_counts, masks = get_blur_np(img_np, labels_np, size, blur=False)
  for mask, rgb in zip(masks, rgb_means):
    for jj in range(3):
      img_np[:, jj][mask] = rgb[jj]

  img_out = rearrange(np.clip(img_np, 0, 255), 'n c h w -> h w (n c)').astype(np.uint8)
  labels_out = labels_np.squeeze().astype(np.uint32)
  return img_out, labels_out



 


# use in-file calc_ciede and get_cls_update defined below


def get_blur_torch(img: torch.Tensor, labels: torch.Tensor, size, blur=True):
    if blur:
        assert size % 2 == 1
        p = (size - 1) // 2
        img = F.pad(img, [p, p, p, p], mode='reflect')
        img = F.avg_pool2d(img, kernel_size=size, stride=1)
    
    cls = torch.unique(labels).reshape(-1, 1, 1, 1)
    masks = torch.bitwise_and(img[:, [3]] > 127, cls == labels)

    cls_counts = masks.sum(dim=(2, 3), keepdim=True) + 1e-7
    rgb_means = (img[:, :3] * masks).sum(dim=(2, 3), keepdim=True) / cls_counts
    
    rgb_means = rgb_means.squeeze().cpu().tolist()
    cls_list = cls.squeeze().cpu().tolist()
    cls_counts = cls_counts.squeeze().cpu().tolist()
    
    return rgb_means, cls_list, cls_counts, masks


def get_base_torch(img: np.ndarray, loop, cls_num, threshold, size, kmeans_samples=-1, device='cpu'):
  rgb_flatten = cluster_samples = img[..., :3].reshape((-1, 3))
  im_h, im_w = img.shape[:2]

  alpha_mask = np.where(img[..., 3] > 127)
  resampled = False
  if rgb_flatten.shape[0] > len(alpha_mask[0]):
    cluster_samples = img[..., :3][alpha_mask].reshape((-1, 3))
    resampled = True

  if len(rgb_flatten) > kmeans_samples and kmeans_samples > 0:
    cluster_samples = shuffle(cluster_samples, random_state=0, n_samples=kmeans_samples)
    resampled = True

  kmeans = MiniBatchKMeans(n_clusters=cls_num).fit(cluster_samples)
  if resampled:
    labels = kmeans.predict(rgb_flatten)
  else:
    labels = kmeans.labels_

  img_torch = rearrange([img], 'n h w c -> n c h w')
  img_torch = torch.from_numpy(img_torch).to(dtype=torch.float32, device=device)
  labels_torch = torch.from_numpy(labels.reshape((1, 1, im_h, im_w))).to(dtype=torch.float32, device=device)

  assert loop > 0
  img_torch_ori = img_torch.clone()
  for i in range(loop):
    rgb_means, cls_list, cls_counts, masks = get_blur_torch(img_torch, labels_torch, size)
    ciede_df = calc_ciede(rgb_means, cls_list)
    cls2rgb, cls2counts, cls2masks = {}, {}, {}
    for c, rgb, count, mask in zip(cls_list, rgb_means, cls_counts, masks):
      cls2rgb[c] = rgb
      cls2counts[c] = count
      cls2masks[c] = mask[None, ...]
    merge_dict = get_cls_update_counts(ciede_df, threshold, cls2counts)
    tgt2merge, notmerged = {}, set(cls_list)
    for k, v in merge_dict.items():
      if v not in tgt2merge:
        tgt2merge[v] = []
        notmerged.remove(v)
      tgt2merge[v].append(k)
      notmerged.remove(k)
    for k in notmerged:
      tgt2merge[k] = []

    for tgtc, srcc_list in tgt2merge.items():
      mask = cls2masks[tgtc]
      for srcc in srcc_list:
        mask = torch.bitwise_or(mask, cls2masks[srcc])
      labels_torch.masked_fill_(mask, tgtc)
      if i != loop - 1:
        for jj in range(3):
          img_torch[:, jj].masked_fill_(mask[0], cls2rgb[tgtc][jj])

  cls_list_final = torch.unique(labels_torch)
  img_torch = img_torch_ori
  rgb_means, cls_list, cls_counts, masks = get_blur_torch(img_torch, labels_torch, size, blur=False)
  for mask, rgb in zip(masks, rgb_means):
    for jj in range(3):
      img_torch[:, jj][mask] = rgb[jj]

  img_out = rearrange(img_torch.cpu().numpy(), 'n c h w -> h w (n c)')
  img_out = img_out.clip(0, 255).astype(np.uint8)
  labels_out = labels_torch.cpu().numpy().squeeze().astype(np.uint32)
  return img_out, labels_out



 
# use in-file skimage_rgb2lab/df2rgba/rgba2df/hsv2df/rgb2df/mask2df/img_plot/get_foreground defined below

def calc_ciede(mean_list, cls_list):
  cls_no = []
  tgt_no = []
  ciede_list = []
  for i in range(len(mean_list)):
    img_1 = np.array(mean_list[i][:3])
    for j in range(len(mean_list)):
      if i == j:
        continue
      img_2 = np.array(mean_list[j][:3])
      ciede = color.deltaE_ciede2000(skimage_rgb2lab(img_1), skimage_rgb2lab(img_2))[0][0]
      ciede_list.append(ciede)
      cls_no.append(cls_list[i])
      tgt_no.append(cls_list[j])
  ciede_df = pd.DataFrame({"cls_no": cls_no, "tgt_no": tgt_no, "ciede2000": ciede_list})
  return ciede_df

def get_mask_by_cls(df, cls_no):
  mask = df.copy()
  mask.loc[df["label"] != cls_no, ["r","g","b"]] = 0
  mask.loc[df["label"] == cls_no, ["r","g","b"]] = 255
  mask = cv2.cvtColor(df2rgba(mask).astype(np.uint8), cv2.COLOR_RGBA2GRAY)
  return mask

def fill_mean_color(img_df, mask):
  df_img = df2rgba(img_df).astype(np.uint8)
  if len(df_img.shape) == 3:
      mask = np.repeat(mask[:, :, np.newaxis], df_img.shape[-1], axis=-1)
  masked_img = np.where(mask == 0, 0, df_img)
  mean = np.mean(masked_img[mask != 0].reshape(-1, df_img.shape[-1]), axis=0)

  img_df["r"] = mean[0]
  img_df["g"] = mean[1]
  img_df["b"] = mean[2]
  
  return img_df, mean

def get_blur_cls(img, cls, size):
  blur_img = cv2.blur(img, (size, size))
  blur_df = rgba2df(blur_img)
  blur_df["label"] = cls
  img_list = []
  mean_list = []
  cls_list = list(cls.unique())
  for cls_no in tqdm(cls_list):
    mask = get_mask_by_cls(blur_df, cls_no)
    img_df = blur_df.copy()
    img_df.loc[blur_df["label"] != cls_no, ["a"]] = 0 
    img_df, mean = fill_mean_color(img_df, mask)
    df_img = df2rgba(img_df).astype(np.uint8)
    img_list.append(df_img)
    mean_list.append(mean)
  return img_list, mean_list, cls_list

def get_cls_update(ciede_df, df, threshold):
    set_list = [frozenset({cls, tgt}) for cls, tgt in ciede_df[ciede_df['ciede2000'] < threshold][['cls_no', 'tgt_no']].to_numpy()]
    merge_set = []
    while set_list:
        set_a = set_list.pop()
        merged = False
        for i, set_b in enumerate(merge_set):
            if set_a & set_b:
                merge_set[i] |= set_a
                merged = True
                break
        if not merged:
            merge_set.append(set_a)
    merge_dict = {}
    for merge in merge_set:
        cls_counts = {cls: len(df[df['label'] == cls]) for cls in merge}
        # Pick class with maximum count using explicit key function
        max_cls = max(cls_counts.keys(), key=lambda k: cls_counts[k])
        for cls in merge:
            merge_dict[cls] = max_cls
    return merge_dict


def get_color_dict(mean_list, cls_list):
  color_dict = {}
  for idx, mean in enumerate(mean_list):
    color_dict.update({cls_list[idx]:{"r":mean[0],"g":mean[1],"b":mean[2], }})
  return color_dict

def get_update_df(df, merge_dict, mean_list, cls_list):
  update_df = df.copy()
  update_df["label"] = update_df["label"].apply(lambda x: x if x not in merge_dict.keys() else merge_dict[x])
  color_dict = get_color_dict(mean_list, cls_list)
  update_df["r"] = update_df.apply(lambda x: color_dict[x["label"]]["r"], axis=1)
  update_df["g"] = update_df.apply(lambda x: color_dict[x["label"]]["g"], axis=1)
  update_df["b"] = update_df.apply(lambda x: color_dict[x["label"]]["b"], axis=1)    
  return update_df, color_dict

def split_img_df(df, show=False):
  img_list = []
  for cls_no in tqdm(list(df["label"].unique())):
    img_df = df.copy()
    img_df.loc[df["label"] != cls_no, ["a"]] = 0 
    df_img = df2rgba(img_df).astype(np.uint8)
    img_list.append(df_img)
  return img_list


def get_base(img, loops, cls_num, threshold, size, h_split, v_split, n_cluster, alpha, th_rate, bg_split=True, debug=False):
  if bg_split == False:
    df = rgba2df(img)
    df_list = [df]
  else:
    df_list = get_foreground(img, h_split, v_split, n_cluster, alpha, th_rate)

  output_list = []

  for idx, df in enumerate(df_list):
    output_df = df.copy()
    cls = MiniBatchKMeans(n_clusters = cls_num)
    cls.fit(df[["r","g","b"]])
    df["label"] = cls.labels_ 
    df["label"] = df["label"].astype(str) + f"_{idx}"
    for i in range(loops):
      if i !=0:
        img = df2rgba(df).astype(np.uint8)
      blur_list, mean_list, cls_list = get_blur_cls(img, df["label"], size)
      ciede_df = calc_ciede(mean_list, cls_list)
      merge_dict = get_cls_update(ciede_df, df, threshold)
      update_df, color_dict = get_update_df(df, merge_dict, mean_list, cls_list)
      df = update_df
      if debug==True:
        img_plot(df)
    output_df["label"] = df["label"]
    output_df["layer_no"] = idx 
    output_list.append(output_df)

  output_df = pd.concat(output_list).sort_index()

  mean_list = []
  cls_list = list(output_df["label"].unique())
  for cls_no in tqdm(cls_list):
    mask = get_mask_by_cls(output_df, cls_no)
    img_df = output_df.copy()
    img_df.loc[output_df["label"] != cls_no, ["a"]] = 0 
    img_df, mean = fill_mean_color(img_df, mask)
    mean_list.append(mean)

  color_dict = get_color_dict(mean_list, cls_list)
  output_df["r"] = output_df.apply(lambda x: color_dict[x["label"]]["r"], axis=1)
  output_df["g"] = output_df.apply(lambda x: color_dict[x["label"]]["g"], axis=1)
  output_df["b"] = output_df.apply(lambda x: color_dict[x["label"]]["b"], axis=1)  
  
  return output_df

def set_label(x, idx):
  if x["m_flg"] == True:
    return idx
  else :
    return x["label"]

def mode_fast(series):
    return series.mode().iloc[0]

def get_seg_base(input_image, masks, th):
  df = rgba2df(input_image)
  df["label"] = -1
  for idx, mask in tqdm(enumerate(masks)):
    if int(mask["area"] < th):
      continue
    mask_df = mask2df(mask["segmentation"])
    # 左外部結合で全画素の座標を保持しつつ、このマスクが当たるピクセルのみ m_flg=True を付与
    df = df.merge(mask_df, left_on=["x_l", "y_l"], right_on=["x_l_m", "y_l_m"], how="left")
    # 欠損は False 扱いにしてブール化
    df["m_flg"] = df["m_flg"].fillna(0).astype(bool)
    df["label"] = np.where(df["m_flg"], idx, df["label"])
    df.drop(columns=["x_l_m", "y_l_m", "m_flg"], inplace=True)

  df['r'] = df.groupby('label')['r'].transform(mode_fast)
  df['g'] = df.groupby('label')['g'].transform(mode_fast)
  df['b'] = df.groupby('label')['b'].transform(mode_fast)
  return df

def get_normal_layer(input_image, df):
  base_layer_list = split_img_df(df, show=False)

  org_df = rgba2df(input_image)
  hsv_df = hsv2df(cv2.cvtColor(df2rgba(df).astype(np.uint8), cv2.COLOR_RGB2HSV))
  hsv_org = hsv2df(cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV))

  hsv_org["bright_flg"] = hsv_df["v"] < hsv_org["v"]
  bright_df = org_df.copy()
  bright_df["bright_flg"] = hsv_org["bright_flg"]
  bright_df["a"] = np.where(bright_df["bright_flg"] == True, 255, 0)
  bright_df["label"] = df["label"]
  bright_layer_list = split_img_df(bright_df, show=False)

  hsv_org["shadow_flg"] = hsv_df["v"] >= hsv_org["v"]
  shadow_df = rgba2df(input_image)
  shadow_df["shadow_flg"] = hsv_org["shadow_flg"]
  shadow_df["a"] = np.where(shadow_df["shadow_flg"] == True, 255, 0)
  shadow_df["label"] = df["label"]
  shadow_layer_list = split_img_df(shadow_df, show=True)
    
  return base_layer_list, bright_layer_list, shadow_layer_list


def get_composite_layer(input_image, df):
  base_layer_list = split_img_df(df, show=False)

  org_df = rgba2df(input_image)

  org_df["r"] = org_df["r"].apply(lambda x:int(x))
  org_df["g"] = org_df["g"].apply(lambda x:int(x))
  org_df["b"] = org_df["b"].apply(lambda x:int(x))

  org_df["diff_r"] = df["r"] - org_df["r"]
  org_df["diff_g"] = df["g"] - org_df["g"]
  org_df["diff_b"] = df["b"] - org_df["b"]
  
  org_df["shadow_flg"] = org_df.apply(
    lambda x: True if x["diff_r"] >= 0 and x["diff_g"] >= 0 and x["diff_b"] >= 0 else False,
    axis=1
  )
  org_df["screen_flg"] = org_df.apply(
    lambda x: True if x["diff_r"] < 0 and x["diff_g"] < 0 and x["diff_b"] < 0 else False,
    axis=1
  )
    

  shadow_df = org_df.copy()
  shadow_df["a"] = org_df.apply(lambda x: 255 if x["shadow_flg"] == True else 0, axis=1)
  
  shadow_df["r"] = shadow_df["r"].apply(lambda x: x*255)
  shadow_df["g"] = shadow_df["g"].apply(lambda x: x*255)
  shadow_df["b"] = shadow_df["b"].apply(lambda x: x*255)

  shadow_df["r"] = (shadow_df["r"])/df["r"]
  shadow_df["g"] = (shadow_df["g"])/df["g"]
  shadow_df["b"] = (shadow_df["b"])/df["b"]
  
  shadow_df["label"] = df["label"]
  shadow_layer_list = split_img_df(shadow_df, show=True)

  screen_df = org_df.copy()

  screen_df["a"] = screen_df["screen_flg"].apply(lambda x: 255 if x == True else 0)

  screen_df["r"] = (screen_df["r"] - df["r"])/(1 - df["r"]/255) 
  screen_df["g"] = (screen_df["g"] - df["g"])/(1 - df["g"]/255) 
  screen_df["b"] = (screen_df["b"] - df["b"])/(1 - df["b"]/255) 

  screen_df["label"] = df["label"]
  screen_layer_list = split_img_df(screen_df, show=True)

  
  addition_df = org_df.copy()
  addition_df["a"] = addition_df.apply(lambda x: 255 if x["screen_flg"] == False and x["shadow_flg"] == False else 0, axis=1)

  addition_df["r"] = org_df["r"] - df["r"] 
  addition_df["g"] = org_df["g"] - df["g"] 
  addition_df["b"] = org_df["b"] - df["b"]  

  addition_df["r"] = addition_df["r"].apply(lambda x: 0 if x < 0 else x)
  addition_df["g"] = addition_df["g"].apply(lambda x: 0 if x < 0 else x)
  addition_df["b"] = addition_df["b"].apply(lambda x: 0 if x < 0 else x)

  addition_df["label"] = df["label"]

  addition_layer_list = split_img_df(addition_df, show=True)

  subtract_df = org_df.copy()
  subtract_df["a"] = subtract_df.apply(lambda x: 255 if x["screen_flg"] == False and x["shadow_flg"] == False else 0, axis=1)

  subtract_df["r"] = df["r"] - org_df["r"]   
  subtract_df["g"] = df["g"] - org_df["g"] 
  subtract_df["b"] = df["b"] - org_df["b"]

  subtract_df["r"] = subtract_df["r"].apply(lambda x: 0 if x < 0 else x)
  subtract_df["g"] = subtract_df["g"].apply(lambda x: 0 if x < 0 else x)
  subtract_df["b"] = subtract_df["b"].apply(lambda x: 0 if x < 0 else x)

  subtract_df["label"] = df["label"]

  subtract_layer_list = split_img_df(subtract_df, show=True)


  return base_layer_list, shadow_layer_list, screen_layer_list, addition_layer_list, subtract_layer_list


 

def get_mask_generator(pred_iou_thresh, stability_score_thresh, min_mask_region_area, model_path, exe_mode):
  sam_checkpoint = os.path.join(model_path, "sam_vit_h_4b8939.pth")
  device = "cuda"
  model_type = "default"

  if exe_mode == "extension":
    try:
      from modules.safe import unsafe_torch_load, load  # optional environment
      torch.load = unsafe_torch_load
      sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
      sam.to(device=device)
      torch.load = load
    except Exception:
      # fallback to standard load if extension environment not available
      sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
      sam.to(device=device)
  else:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

  mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=pred_iou_thresh,
    stability_score_thresh=stability_score_thresh,
    min_mask_region_area=min_mask_region_area,
  )

  return mask_generator

def get_masks(image, mask_generator):
    masks = mask_generator.generate(image)
    return masks

def show_anns(image, masks, output_dir):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # pickle化してファイルに書き込み
    with open(f'{output_dir}/tmp/seg_layer/sorted_masks.pkl', 'wb') as f:
        pickle.dump(sorted_masks, f)
    polygons = []
    color = []
    mask_list = []
    for mask in sorted_masks:
        m = mask['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        img = np.dstack((img*255, m*255*0.35))
        img = img.astype(np.uint8)

        mask_list.append(img)

    base_mask = image
    for mask in mask_list:
        base_mask = Image.alpha_composite(base_mask, Image.fromarray(mask))

    return base_mask

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)




# using in-file df2rgba

 
def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)


def img_plot(df):
  img = df2rgba(df).astype(np.uint8)
  plt.imshow(img)
  plt.show()


def add_psd(psd, img, name, mode):

  layer_1 = layers.ChannelImageData(image=img[:, :, 3], compression=1)
  layer0 = layers.ChannelImageData(image=img[:, :, 0], compression=1)
  layer1 = layers.ChannelImageData(image=img[:, :, 1], compression=1)
  layer2 = layers.ChannelImageData(image=img[:, :, 2], compression=1)

  new_layer = layers.LayerRecord(channels={-1: layer_1, 0: layer0, 1: layer1, 2: layer2},
                                  top=0, bottom=img.shape[0], left=0, right=img.shape[1],
                                  blend_mode=mode,
                                  name=name,
                                  opacity=255,
                                  )
  #gp = nested_layers.Group()
  #gp.layers = [new_layer]
  psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
  return psd

def load_seg_model(model_dir):
  folder = model_dir
  file_name = 'sam_vit_h_4b8939.pth'
  url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)



def load_masks(output_dir):
  pkl_path = os.path.join(output_dir, "tmp", "seg_layer", "sorted_masks.pkl")
  with open(pkl_path, 'rb') as f:
    masks = pickle.load(f)
  return masks

def save_psd(input_image, layers, names, modes, output_dir, layer_mode):
  psd = pytoshop.core.PsdFile(num_channels=3, height=input_image.shape[0], width=input_image.shape[1])
  if layer_mode == "normal":
    for idx, output in enumerate(layers[0]):
      psd = add_psd(psd, layers[0][idx], names[0] + str(idx), modes[0])
      psd = add_psd(psd, layers[1][idx], names[1] + str(idx), modes[1])
      psd = add_psd(psd, layers[2][idx], names[2] + str(idx), modes[2])
  else:
    for idx, output in enumerate(layers[0]):
      psd = add_psd(psd, layers[0][idx], names[0] + str(idx), modes[0])
      psd = add_psd(psd, layers[1][idx], names[1] + str(idx), modes[1])
      psd = add_psd(psd, layers[2][idx], names[2] + str(idx), modes[2])
      psd = add_psd(psd, layers[3][idx], names[3] + str(idx), modes[3])
      psd = add_psd(psd, layers[4][idx], names[4] + str(idx), modes[4])

  name = randomname(10)

  with open(f"{output_dir}/output_{name}.psd", 'wb') as fd2:
      psd.write(fd2)

  return f"{output_dir}/output_{name}.psd"

def divide_folder(psd_path, input_dir, mode):
  with open(f'{input_dir}/empty.psd', "rb") as fd:
    psd_base = PSD.read(fd)
  with open(psd_path, "rb") as fd:
    psd_image = PSD.read(fd)

  if mode == "normal":
     add_num = 3
  else:
     add_num = 5

  base_records_list = list(psd_base.layer_and_mask_information.layer_info.layer_records)
  image_records_list = list(psd_image.layer_and_mask_information.layer_info.layer_records)

  merge_list = []
  for idx, record in enumerate(image_records_list):
      if idx % add_num == 0:
          merge_list.append(base_records_list[0])
      merge_list.append(record)
      if idx % add_num == (add_num - 1):
          merge_list.append(base_records_list[2])

  psd_image.layer_and_mask_information.layer_info.layer_records = psd_tools.psd.layer_and_mask.LayerRecords(merge_list)
  psd_image.layer_and_mask_information.layer_info.layer_count = len(psd_image.layer_and_mask_information.layer_info.layer_records)

  folder_channel = psd_base.layer_and_mask_information.layer_info.channel_image_data[0]
  image_channel = psd_image.layer_and_mask_information.layer_info.channel_image_data

  channel_list = []
  for idx, channel in enumerate(image_channel):
      if idx % add_num == 0:
          channel_list.append(folder_channel)
      channel_list.append(channel)
      if idx % add_num == (add_num - 1):
          channel_list.append(folder_channel)

  psd_image.layer_and_mask_information.layer_info.channel_image_data =  psd_tools.psd.layer_and_mask.ChannelImageData(channel_list)
  with open(psd_path, 'wb') as fd:
      psd_image.write(fd)

  return psd_path

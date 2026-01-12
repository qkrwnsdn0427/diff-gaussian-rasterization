/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/forward.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug,
	const torch::Tensor& tile_mask,
	const torch::Tensor& gaussian_mask,
	const torch::Tensor& gaussian_mask_tile,
	const int tile_mask_mode,
	const int tile_mask_pad,
	const bool tile_mask_build,
	const bool tile_mask_cull)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int tile_h = (H + BLOCK_Y - 1) / BLOCK_Y;
  const int tile_w = (W + BLOCK_X - 1) / BLOCK_X;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  const int* tile_mask_ptr = nullptr;
	  if (tile_mask.numel() > 0)
	  {
		  if (!tile_mask.is_cuda())
			  AT_ERROR("tile_mask must be a CUDA tensor");
		  if (tile_mask.dim() != 2 || tile_mask.size(0) != tile_h || tile_mask.size(1) != tile_w)
			  AT_ERROR("tile_mask must have shape (ceil(H/BLOCK_Y), ceil(W/BLOCK_X))");
		  if (tile_mask.scalar_type() != at::kInt)
			  AT_ERROR("tile_mask must have dtype int32");
		  tile_mask_ptr = tile_mask.contiguous().data_ptr<int>();
	  }

	  const uint8_t* gaussian_mask_ptr = nullptr;
	  const uint8_t* gaussian_mask_tile_ptr = nullptr;
	  if (gaussian_mask.numel() > 0)
	  {
		  if (!gaussian_mask.is_cuda())
			  AT_ERROR("gaussian_mask must be a CUDA tensor");
		  if (gaussian_mask.dim() != 1 || gaussian_mask.size(0) != P)
			  AT_ERROR("gaussian_mask must have shape (P,)");
		  if (gaussian_mask.scalar_type() != at::kByte && gaussian_mask.scalar_type() != at::kBool)
			  AT_ERROR("gaussian_mask must have dtype uint8 or bool");
		  gaussian_mask_ptr = gaussian_mask.contiguous().data_ptr<uint8_t>();
	  }

	  if (gaussian_mask_tile.numel() > 0)
	  {
		  if (!gaussian_mask_tile.is_cuda())
			  AT_ERROR("gaussian_mask_tile must be a CUDA tensor");
		  if (gaussian_mask_tile.dim() != 1 || gaussian_mask_tile.size(0) != P)
			  AT_ERROR("gaussian_mask_tile must have shape (P,)");
		  if (gaussian_mask_tile.scalar_type() != at::kByte && gaussian_mask_tile.scalar_type() != at::kBool)
			  AT_ERROR("gaussian_mask_tile must have dtype uint8 or bool");
		  gaussian_mask_tile_ptr = gaussian_mask_tile.contiguous().data_ptr<uint8_t>();
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		tile_mask_ptr,
		gaussian_mask_ptr,
		gaussian_mask_tile_ptr,
		tile_mask_mode,
		tile_mask_pad,
		tile_mask_build,
		tile_mask_cull,
		out_color.contiguous().data<float>(),
		out_invdepthptr,
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug,
	const torch::Tensor& tile_mask,
	const torch::Tensor& gaussian_mask)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  const int tile_h = (H + BLOCK_Y - 1) / BLOCK_Y;
  const int tile_w = (W + BLOCK_X - 1) / BLOCK_X;
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
	  const int* tile_mask_ptr = nullptr;
	  const uint8_t* gaussian_mask_ptr = nullptr;
	  if (tile_mask.numel() > 0)
	  {
		  if (!tile_mask.is_cuda())
			  AT_ERROR("tile_mask must be a CUDA tensor");
		  if (tile_mask.dim() != 2 || tile_mask.size(0) != tile_h || tile_mask.size(1) != tile_w)
			  AT_ERROR("tile_mask must have shape (ceil(H/BLOCK_Y), ceil(W/BLOCK_X))");
		  if (tile_mask.scalar_type() != at::kInt)
			  AT_ERROR("tile_mask must have dtype int32");
		  tile_mask_ptr = tile_mask.contiguous().data_ptr<int>();
	  }

	  if (gaussian_mask.numel() > 0)
	  {
		  if (!gaussian_mask.is_cuda())
			  AT_ERROR("gaussian_mask must be a CUDA tensor");
		  if (gaussian_mask.dim() != 1 || gaussian_mask.size(0) != P)
			  AT_ERROR("gaussian_mask must have shape (P,)");
		  if (gaussian_mask.scalar_type() != at::kByte && gaussian_mask.scalar_type() != at::kBool)
			  AT_ERROR("gaussian_mask must have dtype uint8 or bool");
		  gaussian_mask_ptr = gaussian_mask.contiguous().data_ptr<uint8_t>();
	  }

	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  tile_mask_ptr,
	  gaussian_mask_ptr,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor ComputeTileMaskCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float scale_modifier,
	const torch::Tensor& gaussian_mask,
	const int tile_size,
	const int pad_tiles,
	const int mode)
{
  if (!means3D.is_cuda())
	  AT_ERROR("means3D must be a CUDA tensor");
  if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	  AT_ERROR("means3D must have dimensions (num_points, 3)");
  if (!scales.is_cuda() || scales.ndimension() != 2 || scales.size(1) != 3)
	  AT_ERROR("scales must be a CUDA tensor with shape (num_points, 3)");
  if (!rotations.is_cuda() || rotations.ndimension() != 2 || rotations.size(1) != 4)
	  AT_ERROR("rotations must be a CUDA tensor with shape (num_points, 4)");
  if (!viewmatrix.is_cuda() || viewmatrix.numel() != 16)
	  AT_ERROR("viewmatrix must be a CUDA tensor with 16 elements");
  if (!projmatrix.is_cuda() || projmatrix.numel() != 16)
	  AT_ERROR("projmatrix must be a CUDA tensor with 16 elements");
  if (tile_size != BLOCK_X || tile_size != BLOCK_Y)
	  AT_ERROR("tile_size must match BLOCK_X/BLOCK_Y");
  if (mode != 0 && mode != 1)
	  AT_ERROR("mode must be 0 (footprint) or 1 (center)");

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int tile_h = (H + BLOCK_Y - 1) / BLOCK_Y;
  const int tile_w = (W + BLOCK_X - 1) / BLOCK_X;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  torch::Tensor tile_mask = torch::zeros({tile_h, tile_w}, int_opts);

  const uint8_t* gaussian_mask_ptr = nullptr;
  if (gaussian_mask.numel() > 0)
  {
	  if (!gaussian_mask.is_cuda())
		  AT_ERROR("gaussian_mask must be a CUDA tensor");
	  if (gaussian_mask.dim() != 1 || gaussian_mask.size(0) != P)
		  AT_ERROR("gaussian_mask must have shape (P,)");
	  if (gaussian_mask.scalar_type() != at::kByte && gaussian_mask.scalar_type() != at::kBool)
		  AT_ERROR("gaussian_mask must have dtype uint8 or bool");
	  gaussian_mask_ptr = gaussian_mask.contiguous().data_ptr<uint8_t>();
  }

  if (P != 0)
  {
	  FORWARD::computeTileMask(
		  P,
		  means3D.contiguous().data<float>(),
		  (glm::vec3*)scales.contiguous().data_ptr<float>(),
		  scale_modifier,
		  (glm::vec4*)rotations.contiguous().data_ptr<float>(),
		  viewmatrix.contiguous().data<float>(),
		  projmatrix.contiguous().data<float>(),
		  tan_fovx,
		  tan_fovy,
		  W,
		  H,
		  gaussian_mask_ptr,
		  pad_tiles,
		  mode,
		  tile_mask.contiguous().data<int>());
  }

  return tile_mask.to(torch::kUInt8);
}

torch::Tensor ComputeGaussianMaskFromTilesCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float scale_modifier,
	const torch::Tensor& tile_mask,
	const torch::Tensor& gaussian_mask,
	const int tile_size,
	const int pad_tiles,
	const int mode)
{
  if (!means3D.is_cuda())
	  AT_ERROR("means3D must be a CUDA tensor");
  if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	  AT_ERROR("means3D must have dimensions (num_points, 3)");
  if (!scales.is_cuda() || scales.ndimension() != 2 || scales.size(1) != 3)
	  AT_ERROR("scales must be a CUDA tensor with shape (num_points, 3)");
  if (!rotations.is_cuda() || rotations.ndimension() != 2 || rotations.size(1) != 4)
	  AT_ERROR("rotations must be a CUDA tensor with shape (num_points, 4)");
  if (!viewmatrix.is_cuda() || viewmatrix.numel() != 16)
	  AT_ERROR("viewmatrix must be a CUDA tensor with 16 elements");
  if (!projmatrix.is_cuda() || projmatrix.numel() != 16)
	  AT_ERROR("projmatrix must be a CUDA tensor with 16 elements");
  if (!tile_mask.is_cuda())
	  AT_ERROR("tile_mask must be a CUDA tensor");
  if (tile_mask.dim() != 2)
	  AT_ERROR("tile_mask must have shape (tile_h, tile_w)");
  if (tile_mask.scalar_type() != at::kInt)
	  AT_ERROR("tile_mask must have dtype int32");
  if (tile_size != BLOCK_X || tile_size != BLOCK_Y)
	  AT_ERROR("tile_size must match BLOCK_X/BLOCK_Y");
  if (mode != 0 && mode != 1)
	  AT_ERROR("mode must be 0 (footprint) or 1 (center)");

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int tile_h = (H + BLOCK_Y - 1) / BLOCK_Y;
  const int tile_w = (W + BLOCK_X - 1) / BLOCK_X;
  if (tile_mask.size(0) != tile_h || tile_mask.size(1) != tile_w)
	  AT_ERROR("tile_mask has incorrect shape for image size");

  auto out_mask = torch::zeros({P}, means3D.options().dtype(torch::kUInt8));

  const uint8_t* gaussian_mask_ptr = nullptr;
  if (gaussian_mask.numel() > 0)
  {
	  if (!gaussian_mask.is_cuda())
		  AT_ERROR("gaussian_mask must be a CUDA tensor");
	  if (gaussian_mask.dim() != 1 || gaussian_mask.size(0) != P)
		  AT_ERROR("gaussian_mask must have shape (P,)");
	  if (gaussian_mask.scalar_type() != at::kByte && gaussian_mask.scalar_type() != at::kBool)
		  AT_ERROR("gaussian_mask must have dtype uint8 or bool");
	  gaussian_mask_ptr = gaussian_mask.contiguous().data_ptr<uint8_t>();
  }

  if (P != 0 && tile_mask.numel() > 0)
  {
	  FORWARD::computeGaussianMaskFromTiles(
		  P,
		  means3D.contiguous().data<float>(),
		  (glm::vec3*)scales.contiguous().data_ptr<float>(),
		  scale_modifier,
		  (glm::vec4*)rotations.contiguous().data_ptr<float>(),
		  viewmatrix.contiguous().data<float>(),
		  projmatrix.contiguous().data<float>(),
		  tan_fovx,
		  tan_fovy,
		  W,
		  H,
		  tile_mask.contiguous().data<int>(),
		  gaussian_mask_ptr,
		  pad_tiles,
		  mode,
		  out_mask.contiguous().data_ptr<uint8_t>());
  }

  return out_mask;
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

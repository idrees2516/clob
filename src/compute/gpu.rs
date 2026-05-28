use ocl::{ProQue, Buffer, Program, Kernel, Queue, Event};
use thiserror::Error;
use std::sync::Arc;

#[derive(Debug, Error)]
pub enum GPUError {
    #[error("OpenCL error: {0}")]
    OpenCLError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

pub struct GPUAccelerator {
    pro_que: ProQue,
    max_work_group_size: usize,
}

impl GPUAccelerator {
    pub fn new() -> Result<Self, GPUError> {
        let src = r#"
            __kernel void compute_covariance(
                __global const float* x,
                __global const float* y,
                __global float* result,
                const unsigned int n
            ) {
                const unsigned int gid = get_global_id(0);
                const unsigned int local_size = get_local_size(0);
                const unsigned int group_id = get_group_id(0);
                
                __local float local_sum[256];
                float sum = 0.0f;
                
                for (unsigned int i = gid; i < n; i += get_global_size(0)) {
                    sum += x[i] * y[i];
                }
                
                local_sum[get_local_id(0)] = sum;
                barrier(CLK_LOCAL_MEM_FENCE);
                
                for (unsigned int s = local_size/2; s > 0; s >>= 1) {
                    if (get_local_id(0) < s) {
                        local_sum[get_local_id(0)] += local_sum[get_local_id(0) + s];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                
                if (get_local_id(0) == 0) {
                    result[group_id] = local_sum[0];
                }
            }

            __kernel void compute_volatility(
                __global const float* returns,
                __global float* volatility,
                const unsigned int window_size,
                const unsigned int n
            ) {
                const unsigned int gid = get_global_id(0);
                if (gid >= n - window_size + 1) return;
                
                float sum = 0.0f;
                float sum_sq = 0.0f;
                
                for (unsigned int i = 0; i < window_size; i++) {
                    float r = returns[gid + i];
                    sum += r;
                    sum_sq += r * r;
                }
                
                float mean = sum / window_size;
                float variance = (sum_sq - sum * mean) / (window_size - 1);
                volatility[gid] = sqrt(variance);
            }

            __kernel void compute_spread_estimator(
                __global const float* prices,
                __global const float* volumes,
                __global float* spreads,
                const unsigned int window_size,
                const unsigned int n
            ) {
                const unsigned int gid = get_global_id(0);
                if (gid >= n - window_size + 1) return;
                
                float price_sum = 0.0f;
                float volume_sum = 0.0f;
                float weighted_price_sum = 0.0f;
                
                for (unsigned int i = 0; i < window_size; i++) {
                    float price = prices[gid + i];
                    float volume = volumes[gid + i];
                    price_sum += price;
                    volume_sum += volume;
                    weighted_price_sum += price * volume;
                }
                
                float vwap = weighted_price_sum / volume_sum;
                float mean_price = price_sum / window_size;
                float spread = 2.0f * fabs(vwap - mean_price);
                spreads[gid] = spread;
            }
        "#;

        let pro_que = ProQue::builder()
            .src(src)
            .dims(1)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let max_work_group_size = pro_que.device()
            .max_work_group_size()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        Ok(Self {
            pro_que,
            max_work_group_size,
        })
    }

    pub fn compute_covariance_matrix(
        &self,
        data: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GPUError> {
        let n_vars = data.len();
        if n_vars == 0 {
            return Err(GPUError::InvalidParameters("Empty data".to_string()));
        }
        let n_obs = data[0].len();

        let mut result = vec![vec![0.0f32; n_vars]; n_vars];
        
        for i in 0..n_vars {
            for j in i..n_vars {
                let cov = self.compute_covariance(&data[i], &data[j])?;
                result[i][j] = cov;
                result[j][i] = cov;
            }
        }

        Ok(result)
    }

    fn compute_covariance(
        &self,
        x: &[f32],
        y: &[f32],
    ) -> Result<f32, GPUError> {
        let n = x.len();
        if n != y.len() {
            return Err(GPUError::InvalidParameters(
                "Vectors must have same length".to_string(),
            ));
        }

        let buffer_x = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n)
            .copy_host_slice(x)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let buffer_y = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n)
            .copy_host_slice(y)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let n_work_groups = (n + self.max_work_group_size - 1) / self.max_work_group_size;
        let result_buffer = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().write_only())
            .len(n_work_groups)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let kernel = self.pro_que.kernel_builder("compute_covariance")
            .arg(&buffer_x)
            .arg(&buffer_y)
            .arg(&result_buffer)
            .arg(n as u32)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        unsafe {
            kernel.enq()
                .map_err(|e| GPUError::OpenCLError(e.to_string()))?;
        }

        let mut result = vec![0.0f32; n_work_groups];
        result_buffer.read(&mut result)
            .enq()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        Ok(result.iter().sum())
    }

    pub fn compute_rolling_volatility(
        &self,
        returns: &[f32],
        window_size: usize,
    ) -> Result<Vec<f32>, GPUError> {
        let n = returns.len();
        if n < window_size {
            return Err(GPUError::InvalidParameters(
                "Window size larger than data length".to_string(),
            ));
        }

        let buffer_returns = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n)
            .copy_host_slice(returns)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let buffer_volatility = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().write_only())
            .len(n - window_size + 1)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let kernel = self.pro_que.kernel_builder("compute_volatility")
            .arg(&buffer_returns)
            .arg(&buffer_volatility)
            .arg(window_size as u32)
            .arg(n as u32)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        unsafe {
            kernel.enq()
                .map_err(|e| GPUError::OpenCLError(e.to_string()))?;
        }

        let mut result = vec![0.0f32; n - window_size + 1];
        buffer_volatility.read(&mut result)
            .enq()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        Ok(result)
    }

    pub fn compute_spread_estimates(
        &self,
        prices: &[f32],
        volumes: &[f32],
        window_size: usize,
    ) -> Result<Vec<f32>, GPUError> {
        let n = prices.len();
        if n != volumes.len() {
            return Err(GPUError::InvalidParameters(
                "Price and volume vectors must have same length".to_string(),
            ));
        }
        if n < window_size {
            return Err(GPUError::InvalidParameters(
                "Window size larger than data length".to_string(),
            ));
        }

        let buffer_prices = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n)
            .copy_host_slice(prices)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let buffer_volumes = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().read_only())
            .len(n)
            .copy_host_slice(volumes)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let buffer_spreads = Buffer::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::MemFlags::new().write_only())
            .len(n - window_size + 1)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        let kernel = self.pro_que.kernel_builder("compute_spread_estimator")
            .arg(&buffer_prices)
            .arg(&buffer_volumes)
            .arg(&buffer_spreads)
            .arg(window_size as u32)
            .arg(n as u32)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        unsafe {
            kernel.enq()
                .map_err(|e| GPUError::OpenCLError(e.to_string()))?;
        }

        let mut result = vec![0.0f32; n - window_size + 1];
        buffer_spreads.read(&mut result)
            .enq()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;

        Ok(result)
    }
}

#[cfg(feature = "gpu")]
use crate::config::{ComputeBackend, ComputeConfig};

#[cfg(feature = "gpu")]
pub enum Backend {
    Cpu,
    Gpu(GPUAccelerator),
}

#[cfg(feature = "gpu")]
impl GPUAccelerator {
    pub fn with_config(config: &ComputeConfig) -> Result<Self, GPUError> {
        // Select device if specified
        let builder = if let Some(ref device_name) = config.gpu_device {
            ProQue::builder().device(device_name)
        } else {
            ProQue::builder()
        };
        let pro_que = builder.src("") // Kernel source will be set per kernel
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;
        let max_work_group_size = pro_que.device().max_wg_size().unwrap_or(256);
        Ok(Self { pro_que, max_work_group_size })
    }

    pub fn simulate_rough_volterra(
        &mut self,
        kernel_src: &str,
        dw: &[f32],
        kernel: &[f32],
        mean_reversion: f32,
        theta: f32,
        vol_of_vol: f32,
        dt: f32,
        n: usize,
    ) -> Result<Vec<f32>, GPUError> {
        use ocl::{Buffer, Kernel};
        let pro_que = ProQue::builder()
            .src(kernel_src)
            .dims(n)
            .build()
            .map_err(|e| GPUError::OpenCLError(e.to_string()))?;
        let dw_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).len(n).copy_host_slice(dw).build().unwrap();
        let kernel_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).len(n).copy_host_slice(kernel).build().unwrap();
        let mut volatility = vec![0.0f32; n];
        let volatility_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).len(n).build().unwrap();
        let k = Kernel::builder()
            .program(&pro_que.program())
            .name("simulate_rough_volterra")
            .queue(pro_que.queue().clone())
            .global_work_size(n)
            .arg(&dw_buf)
            .arg(&kernel_buf)
            .arg(mean_reversion)
            .arg(theta)
            .arg(vol_of_vol)
            .arg(dt)
            .arg(&volatility_buf)
            .arg(n as u32)
            .build().unwrap();
        unsafe { k.enq().unwrap(); }
        volatility_buf.read(&mut volatility).enq().unwrap();
        Ok(volatility)
    }

    pub const VOLTERRA_KERNEL_SRC: &'static str = r#"
    __kernel void simulate_rough_volterra(
        __global const float* dw,
        __global const float* kernel,
        float mean_reversion,
        float theta,
        float vol_of_vol,
        float dt,
        __global float* volatility,
        uint n
    ) {
        int i = get_global_id(0);
        if (i >= n) return;
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
            sum += kernel[i - j] * dw[j];
        }
        float prev_vol = (i == 0) ? 0.0f : volatility[i-1];
        float mean_rev = -mean_reversion * (prev_vol - theta);
        volatility[i] = mean_rev * dt + vol_of_vol * sum;
    }
    "#;
}

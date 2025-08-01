//! ethrex RPC Client Implementation
//! 
//! This module provides a concrete implementation of the EthereumRpcClient trait
//! for interacting with ethrex nodes via JSON-RPC.

use crate::rollup::{
    ethrex_client::*,
    types::*,
    RollupError,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, warn, error};

/// ethrex JSON-RPC client implementation
pub struct EthrexRpcClient {
    /// HTTP client for JSON-RPC requests
    client: reqwest::Client,
    /// ethrex node RPC endpoint
    rpc_url: String,
    /// Request timeout duration
    request_timeout: Duration,
    /// Request ID counter for JSON-RPC
    request_id: std::sync::atomic::AtomicU64,
}

/// JSON-RPC request structure
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    params: Value,
    id: u64,
}

/// JSON-RPC response structure
#[derive(Debug, Deserialize)]
struct JsonRpcResponse<T> {
    jsonrpc: String,
    result: Option<T>,
    error: Option<JsonRpcError>,
    id: u64,
}

/// JSON-RPC error structure
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    data: Option<Value>,
}

/// Ethereum block response from RPC
#[derive(Debug, Deserialize)]
struct EthBlockResponse {
    number: String,
    hash: String,
    #[serde(rename = "parentHash")]
    parent_hash: String,
    timestamp: String,
    transactions: Vec<String>,
    #[serde(rename = "gasUsed")]
    gas_used: String,
    #[serde(rename = "gasLimit")]
    gas_limit: String,
}

/// Ethereum transaction response from RPC
#[derive(Debug, Deserialize)]
struct EthTransactionResponse {
    hash: String,
    #[serde(rename = "blockNumber")]
    block_number: Option<String>,
    #[serde(rename = "blockHash")]
    block_hash: Option<String>,
    #[serde(rename = "transactionIndex")]
    transaction_index: Option<String>,
    from: String,
    to: Option<String>,
    value: String,
    gas: String,
    #[serde(rename = "gasPrice")]
    gas_price: String,
    input: String,
    nonce: String,
}

/// Ethereum transaction receipt response from RPC
#[derive(Debug, Deserialize)]
struct EthReceiptResponse {
    #[serde(rename = "transactionHash")]
    transaction_hash: String,
    #[serde(rename = "blockNumber")]
    block_number: String,
    #[serde(rename = "blockHash")]
    block_hash: String,
    #[serde(rename = "transactionIndex")]
    transaction_index: String,
    from: String,
    to: Option<String>,
    #[serde(rename = "gasUsed")]
    gas_used: String,
    status: String,
    logs: Vec<EthLogResponse>,
}

/// Ethereum log response from RPC
#[derive(Debug, Deserialize)]
struct EthLogResponse {
    address: String,
    topics: Vec<String>,
    data: String,
}

impl EthrexRpcClient {
    /// Create a new ethrex RPC client
    pub fn new(rpc_url: String, request_timeout: Duration) -> Result<Self, RollupError> {
        let client = reqwest::Client::builder()
            .timeout(request_timeout)
            .build()
            .map_err(|e| RollupError::NetworkError(e))?;

        Ok(Self {
            client,
            rpc_url,
            request_timeout,
            request_id: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Make a JSON-RPC request to ethrex
    async fn make_rpc_request<T>(&self, method: &str, params: Value) -> Result<T, RollupError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let request_id = self.request_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: request_id,
        };

        debug!("Making RPC request to ethrex: {} with params: {:?}", method, params);

        let response = timeout(
            self.request_timeout,
            self.client.post(&self.rpc_url).json(&request).send()
        )
        .await
        .map_err(|_| RollupError::NetworkError(
            reqwest::Error::from(reqwest::ErrorKind::Request)
        ))?
        .map_err(|e| RollupError::NetworkError(e))?;

        if !response.status().is_success() {
            return Err(RollupError::NetworkError(
                reqwest::Error::from(reqwest::ErrorKind::Request)
            ));
        }

        let rpc_response: JsonRpcResponse<T> = response
            .json()
            .await
            .map_err(|e| RollupError::NetworkError(e))?;

        if let Some(error) = rpc_response.error {
            return Err(RollupError::L1SyncError(format!(
                "RPC error {}: {}",
                error.code, error.message
            )));
        }

        rpc_response.result.ok_or_else(|| {
            RollupError::L1SyncError("RPC response missing result".to_string())
        })
    }

    /// Convert hex string to bytes
    fn hex_to_bytes(&self, hex_str: &str) -> Result<Vec<u8>, RollupError> {
        let hex_str = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        hex::decode(hex_str).map_err(|e| {
            RollupError::L1SyncError(format!("Invalid hex string: {}", e))
        })
    }

    /// Convert hex string to fixed-size array
    fn hex_to_fixed_bytes<const N: usize>(&self, hex_str: &str) -> Result<[u8; N], RollupError> {
        let bytes = self.hex_to_bytes(hex_str)?;
        if bytes.len() != N {
            return Err(RollupError::L1SyncError(format!(
                "Expected {} bytes, got {}",
                N, bytes.len()
            )));
        }
        let mut result = [0u8; N];
        result.copy_from_slice(&bytes);
        Ok(result)
    }

    /// Convert hex string to u64
    fn hex_to_u64(&self, hex_str: &str) -> Result<u64, RollupError> {
        let hex_str = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        u64::from_str_radix(hex_str, 16).map_err(|e| {
            RollupError::L1SyncError(format!("Invalid hex number: {}", e))
        })
    }

    /// Convert bytes to hex string
    fn bytes_to_hex(&self, bytes: &[u8]) -> String {
        format!("0x{}", hex::encode(bytes))
    }
}

#[async_trait::async_trait]
impl EthereumRpcClient for EthrexRpcClient {
    /// Get current block number from ethrex
    async fn get_block_number(&self) -> Result<L1BlockNumber, RollupError> {
        let result: String = self
            .make_rpc_request("eth_blockNumber", json!([]))
            .await?;
        
        self.hex_to_u64(&result)
    }

    /// Get block by number from ethrex
    async fn get_block(&self, block_number: L1BlockNumber) -> Result<EthereumBlock, RollupError> {
        let block_hex = format!("0x{:x}", block_number);
        let result: EthBlockResponse = self
            .make_rpc_request("eth_getBlockByNumber", json!([block_hex, false]))
            .await?;

        Ok(EthereumBlock {
            number: self.hex_to_u64(&result.number)?,
            hash: self.hex_to_fixed_bytes(&result.hash)?,
            parent_hash: self.hex_to_fixed_bytes(&result.parent_hash)?,
            timestamp: self.hex_to_u64(&result.timestamp)?,
            transactions: result.transactions
                .into_iter()
                .map(|tx_hash| self.hex_to_fixed_bytes(&tx_hash))
                .collect::<Result<Vec<_>, _>>()?,
            gas_used: self.hex_to_u64(&result.gas_used)?,
            gas_limit: self.hex_to_u64(&result.gas_limit)?,
        })
    }

    /// Send raw transaction to ethrex
    async fn send_raw_transaction(&self, tx_data: &[u8]) -> Result<TxHash, RollupError> {
        let tx_hex = self.bytes_to_hex(tx_data);
        let result: String = self
            .make_rpc_request("eth_sendRawTransaction", json!([tx_hex]))
            .await?;

        self.hex_to_fixed_bytes(&result)
    }

    /// Get transaction receipt from ethrex
    async fn get_transaction_receipt(&self, tx_hash: TxHash) -> Result<Option<TransactionReceipt>, RollupError> {
        let tx_hex = self.bytes_to_hex(&tx_hash);
        
        // ethrex might return null for pending transactions
        let result: Option<EthReceiptResponse> = self
            .make_rpc_request("eth_getTransactionReceipt", json!([tx_hex]))
            .await?;

        if let Some(receipt) = result {
            let status = match receipt.status.as_str() {
                "0x1" => TransactionStatus::Success,
                "0x0" => TransactionStatus::Failed,
                _ => return Err(RollupError::L1SyncError(
                    format!("Unknown transaction status: {}", receipt.status)
                )),
            };

            let logs = receipt.logs
                .into_iter()
                .map(|log| {
                    Ok(EventLog {
                        address: self.hex_to_fixed_bytes(&log.address)?,
                        topics: log.topics
                            .into_iter()
                            .map(|topic| self.hex_to_fixed_bytes(&topic))
                            .collect::<Result<Vec<_>, _>>()?,
                        data: self.hex_to_bytes(&log.data)?,
                    })
                })
                .collect::<Result<Vec<_>, RollupError>>()?;

            Ok(Some(TransactionReceipt {
                transaction_hash: self.hex_to_fixed_bytes(&receipt.transaction_hash)?,
                block_number: self.hex_to_u64(&receipt.block_number)?,
                block_hash: self.hex_to_fixed_bytes(&receipt.block_hash)?,
                transaction_index: self.hex_to_u64(&receipt.transaction_index)? as u32,
                from: self.hex_to_fixed_bytes(&receipt.from)?,
                to: receipt.to.map(|to| self.hex_to_fixed_bytes(&to)).transpose()?,
                gas_used: self.hex_to_u64(&receipt.gas_used)?,
                status,
                logs,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get transaction by hash from ethrex
    async fn get_transaction(&self, tx_hash: TxHash) -> Result<Option<EthereumTransaction>, RollupError> {
        let tx_hex = self.bytes_to_hex(&tx_hash);
        
        let result: Option<EthTransactionResponse> = self
            .make_rpc_request("eth_getTransactionByHash", json!([tx_hex]))
            .await?;

        if let Some(tx) = result {
            Ok(Some(EthereumTransaction {
                hash: self.hex_to_fixed_bytes(&tx.hash)?,
                block_number: tx.block_number.map(|bn| self.hex_to_u64(&bn)).transpose()?,
                block_hash: tx.block_hash.map(|bh| self.hex_to_fixed_bytes(&bh)).transpose()?,
                transaction_index: tx.transaction_index.map(|ti| self.hex_to_u64(&ti)).transpose()?.map(|x| x as u32),
                from: self.hex_to_fixed_bytes(&tx.from)?,
                to: tx.to.map(|to| self.hex_to_fixed_bytes(&to)).transpose()?,
                value: self.hex_to_u64(&tx.value)?,
                gas: self.hex_to_u64(&tx.gas)?,
                gas_price: self.hex_to_u64(&tx.gas_price)?,
                input: self.hex_to_bytes(&tx.input)?,
                nonce: self.hex_to_u64(&tx.nonce)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Estimate gas for transaction
    async fn estimate_gas(&self, tx_request: &TransactionRequest) -> Result<u64, RollupError> {
        let mut params = json!({});
        
        if let Some(from) = tx_request.from {
            params["from"] = json!(self.bytes_to_hex(&from));
        }
        
        if let Some(to) = tx_request.to {
            params["to"] = json!(self.bytes_to_hex(&to));
        }
        
        if let Some(gas) = tx_request.gas {
            params["gas"] = json!(format!("0x{:x}", gas));
        }
        
        if let Some(gas_price) = tx_request.gas_price {
            params["gasPrice"] = json!(format!("0x{:x}", gas_price));
        }
        
        if let Some(value) = tx_request.value {
            params["value"] = json!(format!("0x{:x}", value));
        }
        
        if let Some(data) = &tx_request.data {
            params["data"] = json!(self.bytes_to_hex(data));
        }

        let result: String = self
            .make_rpc_request("eth_estimateGas", json!([params]))
            .await?;

        self.hex_to_u64(&result)
    }

    /// Get current gas price from ethrex
    async fn get_gas_price(&self) -> Result<u64, RollupError> {
        let result: String = self
            .make_rpc_request("eth_gasPrice", json!([]))
            .await?;

        self.hex_to_u64(&result)
    }

    /// Call contract method (read-only)
    async fn call_contract(&self, call_data: &ContractCall) -> Result<Vec<u8>, RollupError> {
        let mut params = json!({
            "to": self.bytes_to_hex(&call_data.to),
            "data": self.bytes_to_hex(&call_data.data)
        });

        let block_param = if let Some(block_number) = call_data.block_number {
            format!("0x{:x}", block_number)
        } else {
            "latest".to_string()
        };

        let result: String = self
            .make_rpc_request("eth_call", json!([params, block_param]))
            .await?;

        self.hex_to_bytes(&result)
    }
}

/// Factory for creating ethrex RPC clients
pub struct EthrexRpcClientFactory;

impl EthrexRpcClientFactory {
    /// Create a new ethrex RPC client from configuration
    pub fn create_client(config: &EthrexClientConfig) -> Result<Arc<dyn EthereumRpcClient>, RollupError> {
        let timeout = Duration::from_secs(config.transaction_timeout_seconds);
        let client = EthrexRpcClient::new(config.rpc_url.clone(), timeout)?;
        Ok(Arc::new(client))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_hex_conversions() {
        let client = EthrexRpcClient::new(
            "http://localhost:8545".to_string(),
            Duration::from_secs(30)
        ).unwrap();

        // Test hex to bytes
        let bytes = client.hex_to_bytes("0x1234abcd").unwrap();
        assert_eq!(bytes, vec![0x12, 0x34, 0xab, 0xcd]);

        // Test hex to fixed bytes
        let fixed: [u8; 4] = client.hex_to_fixed_bytes("0x1234abcd").unwrap();
        assert_eq!(fixed, [0x12, 0x34, 0xab, 0xcd]);

        // Test hex to u64
        let num = client.hex_to_u64("0x1234").unwrap();
        assert_eq!(num, 0x1234);

        // Test bytes to hex
        let hex = client.bytes_to_hex(&[0x12, 0x34, 0xab, 0xcd]);
        assert_eq!(hex, "0x1234abcd");
    }

    #[test]
    fn test_client_creation() {
        let config = EthrexClientConfig::default();
        let client = EthrexRpcClientFactory::create_client(&config);
        assert!(client.is_ok());
    }

    // Note: Integration tests would require a running ethrex node
    // These would be added in a separate integration test suite
}
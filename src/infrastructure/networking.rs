use std::sync::Arc;
use std::net::SocketAddr;
use std::collections::HashMap;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::{mpsc, RwLock};
use bytes::{Bytes, BytesMut, BufMut};
use serde::{Serialize, Deserialize};
use crate::error::InfrastructureError;

/// Zero-copy networking layer for high-frequency trading
pub struct ZeroCopyNetworking {
    listeners: HashMap<String, TcpListener>,
    connections: Arc<RwLock<HashMap<String, Connection>>>,
    message_pool: Arc<MessagePool>,
    stats: Arc<RwLock<NetworkStats>>,
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub id: String,
    pub addr: SocketAddr,
    pub stream: Arc<tokio::sync::Mutex<TcpStream>>,
    pub buffer_pool: Arc<BufferPool>,
    pub last_activity: std::time::Instant,
}

/// High-performance buffer pool for zero-copy operations
pub struct BufferPool {
    small_buffers: Arc<RwLock<Vec<BytesMut>>>,  // 4KB buffers
    medium_buffers: Arc<RwLock<Vec<BytesMut>>>, // 64KB buffers
    large_buffers: Arc<RwLock<Vec<BytesMut>>>,  // 1MB buffers
    stats: Arc<RwLock<BufferStats>>,
}

#[derive(Debug, Default)]
pub struct BufferStats {
    pub small_allocated: usize,
    pub medium_allocated: usize,
    pub large_allocated: usize,
    pub small_in_use: usize,
    pub medium_in_use: usize,
    pub large_in_use: usize,
}

/// Message pool for object reuse
pub struct MessagePool {
    order_messages: Arc<RwLock<Vec<OrderMessage>>>,
    trade_messages: Arc<RwLock<Vec<TradeMessage>>>,
    market_data_messages: Arc<RwLock<Vec<MarketDataMessage>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderMessage {
    pub order_id: u64,
    pub symbol: String,
    pub side: u8,
    pub price: u64,  // Fixed-point representation
    pub quantity: u64,
    pub timestamp: u64,
    pub client_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeMessage {
    pub trade_id: u64,
    pub symbol: String,
    pub price: u64,
    pub quantity: u64,
    pub timestamp: u64,
    pub buyer_id: String,
    pub seller_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataMessage {
    pub symbol: String,
    pub best_bid: u64,
    pub best_ask: u64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub last_trade_price: u64,
    pub timestamp: u64,
}

#[derive(Debug, Default)]
pub struct NetworkStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub connections_active: usize,
    pub avg_latency_ns: u64,
    pub max_latency_ns: u64,
}

impl ZeroCopyNetworking {
    pub fn new() -> Self {
        Self {
            listeners: HashMap::new(),
            connections: Arc::new(RwLock::new(HashMap::new())),
            message_pool: Arc::new(MessagePool::new()),
            stats: Arc::new(RwLock::new(NetworkStats::default())),
        }
    }

    /// Start listening on specified address with zero-copy optimizations
    pub async fn listen(&mut self, name: String, addr: SocketAddr) -> Result<(), InfrastructureError> {
        let listener = TcpListener::bind(addr).await
            .map_err(|e| InfrastructureError::NetworkError(e.to_string()))?;
        
        // Set socket options for low latency
        self.configure_socket_options(&listener)?;
        
        self.listeners.insert(name.clone(), listener);
        
        // Start accepting connections
        self.start_accept_loop(name).await;
        
        Ok(())
    }

    /// Configure socket for ultra-low latency
    fn configure_socket_options(&self, listener: &TcpListener) -> Result<(), InfrastructureError> {
        use socket2::{Socket, Domain, Type, Protocol};
        
        let socket = Socket::from(listener.as_raw_fd());
        
        // Enable TCP_NODELAY to disable Nagle's algorithm
        socket.set_nodelay(true)
            .map_err(|e| InfrastructureError::NetworkError(e.to_string()))?;
        
        // Set SO_REUSEADDR
        socket.set_reuse_address(true)
            .map_err(|e| InfrastructureError::NetworkError(e.to_string()))?;
        
        // Set receive buffer size
        socket.set_recv_buffer_size(1024 * 1024)  // 1MB
            .map_err(|e| InfrastructureError::NetworkError(e.to_string()))?;
        
        // Set send buffer size
        socket.set_send_buffer_size(1024 * 1024)  // 1MB
            .map_err(|e| InfrastructureError::NetworkError(e.to_string()))?;
        
        Ok(())
    }

    /// Start accepting connections with optimized handling
    async fn start_accept_loop(&self, listener_name: String) {
        let listener = self.listeners.get(&listener_name).unwrap();
        let connections = Arc::clone(&self.connections);
        let message_pool = Arc::clone(&self.message_pool);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let conn_id = format!("{}:{}", addr.ip(), addr.port());
                        let connection = Connection {
                            id: conn_id.clone(),
                            addr,
                            stream: Arc::new(tokio::sync::Mutex::new(stream)),
                            buffer_pool: Arc::new(BufferPool::new()),
                            last_activity: std::time::Instant::now(),
                        };

                        // Add to connections map
                        {
                            let mut conns = connections.write().await;
                            conns.insert(conn_id.clone(), connection.clone());
                        }

                        // Update stats
                        {
                            let mut stats_guard = stats.write().await;
                            stats_guard.connections_active += 1;
                        }

                        // Start handling this connection
                        Self::handle_connection(connection, Arc::clone(&message_pool), Arc::clone(&stats)).await;
                    }
                    Err(e) => {
                        eprintln!("Failed to accept connection: {}", e);
                    }
                }
            }
        });
    }

    /// Handle individual connection with zero-copy message processing
    async fn handle_connection(
        connection: Connection,
        message_pool: Arc<MessagePool>,
        stats: Arc<RwLock<NetworkStats>>,
    ) {
        let stream = Arc::clone(&connection.stream);
        let buffer_pool = Arc::clone(&connection.buffer_pool);

        tokio::spawn(async move {
            let mut stream_guard = stream.lock().await;
            let (reader, writer) = stream_guard.split();
            let mut buf_reader = BufReader::new(reader);
            let mut buf_writer = BufWriter::new(writer);

            let mut read_buffer = buffer_pool.get_medium_buffer().await;

            loop {
                match buf_reader.read_buf(&mut read_buffer).await {
                    Ok(0) => break, // Connection closed
                    Ok(n) => {
                        let start_time = std::time::Instant::now();
                        
                        // Process message without copying
                        if let Err(e) = Self::process_message_zero_copy(
                            &read_buffer[..n],
                            &message_pool,
                            &mut buf_writer,
                        ).await {
                            eprintln!("Error processing message: {}", e);
                        }

                        // Update latency stats
                        let latency_ns = start_time.elapsed().as_nanos() as u64;
                        {
                            let mut stats_guard = stats.write().await;
                            stats_guard.messages_received += 1;
                            stats_guard.bytes_received += n as u64;
                            stats_guard.avg_latency_ns = 
                                (stats_guard.avg_latency_ns + latency_ns) / 2;
                            if latency_ns > stats_guard.max_latency_ns {
                                stats_guard.max_latency_ns = latency_ns;
                            }
                        }

                        read_buffer.clear();
                    }
                    Err(e) => {
                        eprintln!("Error reading from connection: {}", e);
                        break;
                    }
                }
            }

            // Return buffer to pool
            buffer_pool.return_medium_buffer(read_buffer).await;
        });
    }

    /// Process messages with zero-copy techniques
    async fn process_message_zero_copy(
        data: &[u8],
        message_pool: &MessagePool,
        writer: &mut BufWriter<tokio::io::WriteHalf<TcpStream>>,
    ) -> Result<(), InfrastructureError> {
        // Parse message type from first byte
        if data.is_empty() {
            return Ok(());
        }

        match data[0] {
            1 => {
                // Order message
                let mut order_msg = message_pool.get_order_message().await;
                Self::deserialize_order_message(&data[1..], &mut order_msg)?;
                
                // Process order (placeholder)
                println!("Processing order: {:?}", order_msg);
                
                // Return to pool
                message_pool.return_order_message(order_msg).await;
            }
            2 => {
                // Trade message
                let mut trade_msg = message_pool.get_trade_message().await;
                Self::deserialize_trade_message(&data[1..], &mut trade_msg)?;
                
                // Process trade (placeholder)
                println!("Processing trade: {:?}", trade_msg);
                
                // Return to pool
                message_pool.return_trade_message(trade_msg).await;
            }
            3 => {
                // Market data request
                let mut market_data = message_pool.get_market_data_message().await;
                Self::deserialize_market_data_message(&data[1..], &mut market_data)?;
                
                // Send response
                let response = Self::serialize_market_data_response(&market_data)?;
                writer.write_all(&response).await
                    .map_err(|e| InfrastructureError::NetworkError(e.to_string()))?;
                writer.flush().await
                    .map_err(|e| InfrastructureError::NetworkError(e.to_string()))?;
                
                // Return to pool
                message_pool.return_market_data_message(market_data).await;
            }
            _ => {
                return Err(InfrastructureError::InvalidMessage("Unknown message type".to_string()));
            }
        }

        Ok(())
    }

    /// Fast binary deserialization for order messages
    fn deserialize_order_message(data: &[u8], msg: &mut OrderMessage) -> Result<(), InfrastructureError> {
        if data.len() < 32 {
            return Err(InfrastructureError::InvalidMessage("Order message too short".to_string()));
        }

        let mut offset = 0;
        
        // Read order_id (8 bytes)
        msg.order_id = u64::from_le_bytes(
            data[offset..offset+8].try_into()
                .map_err(|_| InfrastructureError::InvalidMessage("Invalid order_id".to_string()))?
        );
        offset += 8;

        // Read symbol length and symbol
        let symbol_len = data[offset] as usize;
        offset += 1;
        if offset + symbol_len > data.len() {
            return Err(InfrastructureError::InvalidMessage("Invalid symbol length".to_string()));
        }
        msg.symbol = String::from_utf8_lossy(&data[offset..offset+symbol_len]).to_string();
        offset += symbol_len;

        // Read remaining fields
        msg.side = data[offset];
        offset += 1;
        
        msg.price = u64::from_le_bytes(
            data[offset..offset+8].try_into()
                .map_err(|_| InfrastructureError::InvalidMessage("Invalid price".to_string()))?
        );
        offset += 8;
        
        msg.quantity = u64::from_le_bytes(
            data[offset..offset+8].try_into()
                .map_err(|_| InfrastructureError::InvalidMessage("Invalid quantity".to_string()))?
        );
        offset += 8;
        
        msg.timestamp = u64::from_le_bytes(
            data[offset..offset+8].try_into()
                .map_err(|_| InfrastructureError::InvalidMessage("Invalid timestamp".to_string()))?
        );

        Ok(())
    }

    /// Fast binary deserialization for trade messages
    fn deserialize_trade_message(data: &[u8], msg: &mut TradeMessage) -> Result<(), InfrastructureError> {
        // Similar implementation to order message
        // Placeholder for brevity
        Ok(())
    }

    /// Fast binary deserialization for market data messages
    fn deserialize_market_data_message(data: &[u8], msg: &mut MarketDataMessage) -> Result<(), InfrastructureError> {
        // Similar implementation to order message
        // Placeholder for brevity
        Ok(())
    }

    /// Serialize market data response
    fn serialize_market_data_response(msg: &MarketDataMessage) -> Result<Vec<u8>, InfrastructureError> {
        let mut response = Vec::with_capacity(64);
        
        // Message type
        response.push(3);
        
        // Symbol
        response.push(msg.symbol.len() as u8);
        response.extend_from_slice(msg.symbol.as_bytes());
        
        // Market data fields
        response.extend_from_slice(&msg.best_bid.to_le_bytes());
        response.extend_from_slice(&msg.best_ask.to_le_bytes());
        response.extend_from_slice(&msg.bid_size.to_le_bytes());
        response.extend_from_slice(&msg.ask_size.to_le_bytes());
        response.extend_from_slice(&msg.last_trade_price.to_le_bytes());
        response.extend_from_slice(&msg.timestamp.to_le_bytes());
        
        Ok(response)
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        self.stats.read().await.clone()
    }
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            small_buffers: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            medium_buffers: Arc::new(RwLock::new(Vec::with_capacity(100))),
            large_buffers: Arc::new(RwLock::new(Vec::with_capacity(10))),
            stats: Arc::new(RwLock::new(BufferStats::default())),
        }
    }

    pub async fn get_small_buffer(&self) -> BytesMut {
        let mut buffers = self.small_buffers.write().await;
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            let mut stats = self.stats.write().await;
            stats.small_allocated += 1;
            BytesMut::with_capacity(4096)
        }
    }

    pub async fn get_medium_buffer(&self) -> BytesMut {
        let mut buffers = self.medium_buffers.write().await;
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            let mut stats = self.stats.write().await;
            stats.medium_allocated += 1;
            BytesMut::with_capacity(65536)
        }
    }

    pub async fn get_large_buffer(&self) -> BytesMut {
        let mut buffers = self.large_buffers.write().await;
        if let Some(buffer) = buffers.pop() {
            buffer
        } else {
            let mut stats = self.stats.write().await;
            stats.large_allocated += 1;
            BytesMut::with_capacity(1048576)
        }
    }

    pub async fn return_small_buffer(&self, mut buffer: BytesMut) {
        buffer.clear();
        let mut buffers = self.small_buffers.write().await;
        if buffers.len() < 1000 {
            buffers.push(buffer);
        }
    }

    pub async fn return_medium_buffer(&self, mut buffer: BytesMut) {
        buffer.clear();
        let mut buffers = self.medium_buffers.write().await;
        if buffers.len() < 100 {
            buffers.push(buffer);
        }
    }

    pub async fn return_large_buffer(&self, mut buffer: BytesMut) {
        buffer.clear();
        let mut buffers = self.large_buffers.write().await;
        if buffers.len() < 10 {
            buffers.push(buffer);
        }
    }
}

impl MessagePool {
    pub fn new() -> Self {
        Self {
            order_messages: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            trade_messages: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            market_data_messages: Arc::new(RwLock::new(Vec::with_capacity(1000))),
        }
    }

    pub async fn get_order_message(&self) -> OrderMessage {
        let mut messages = self.order_messages.write().await;
        messages.pop().unwrap_or_else(|| OrderMessage {
            order_id: 0,
            symbol: String::new(),
            side: 0,
            price: 0,
            quantity: 0,
            timestamp: 0,
            client_id: String::new(),
        })
    }

    pub async fn return_order_message(&self, mut msg: OrderMessage) {
        // Reset message
        msg.order_id = 0;
        msg.symbol.clear();
        msg.side = 0;
        msg.price = 0;
        msg.quantity = 0;
        msg.timestamp = 0;
        msg.client_id.clear();

        let mut messages = self.order_messages.write().await;
        if messages.len() < 10000 {
            messages.push(msg);
        }
    }

    pub async fn get_trade_message(&self) -> TradeMessage {
        let mut messages = self.trade_messages.write().await;
        messages.pop().unwrap_or_else(|| TradeMessage {
            trade_id: 0,
            symbol: String::new(),
            price: 0,
            quantity: 0,
            timestamp: 0,
            buyer_id: String::new(),
            seller_id: String::new(),
        })
    }

    pub async fn return_trade_message(&self, mut msg: TradeMessage) {
        // Reset message
        msg.trade_id = 0;
        msg.symbol.clear();
        msg.price = 0;
        msg.quantity = 0;
        msg.timestamp = 0;
        msg.buyer_id.clear();
        msg.seller_id.clear();

        let mut messages = self.trade_messages.write().await;
        if messages.len() < 10000 {
            messages.push(msg);
        }
    }

    pub async fn get_market_data_message(&self) -> MarketDataMessage {
        let mut messages = self.market_data_messages.write().await;
        messages.pop().unwrap_or_else(|| MarketDataMessage {
            symbol: String::new(),
            best_bid: 0,
            best_ask: 0,
            bid_size: 0,
            ask_size: 0,
            last_trade_price: 0,
            timestamp: 0,
        })
    }

    pub async fn return_market_data_message(&self, mut msg: MarketDataMessage) {
        // Reset message
        msg.symbol.clear();
        msg.best_bid = 0;
        msg.best_ask = 0;
        msg.bid_size = 0;
        msg.ask_size = 0;
        msg.last_trade_price = 0;
        msg.timestamp = 0;

        let mut messages = self.market_data_messages.write().await;
        if messages.len() < 1000 {
            messages.push(msg);
        }
    }
}

// Platform-specific socket configuration
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

#[cfg(windows)]
use std::os::windows::io::AsRawSocket;

#[cfg(unix)]
impl TcpListener {
    fn as_raw_fd(&self) -> std::os::unix::io::RawFd {
        use std::os::unix::io::AsRawFd;
        self.as_raw_fd()
    }
}

#[cfg(windows)]
impl TcpListener {
    fn as_raw_fd(&self) -> std::os::windows::io::RawSocket {
        use std::os::windows::io::AsRawSocket;
        self.as_raw_socket()
    }
}
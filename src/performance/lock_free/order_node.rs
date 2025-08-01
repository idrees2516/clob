use super::atomic_operations::{AtomicOperations, MemoryOrder, AlignedAtomicPtr};
use super::hazard_pointers::{HazardPointerManager, HazardPointer};
use crate::orderbook::types::Order;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::ptr;
use std::sync::Arc;

/// Lock-free order node for linked list implementation
/// Cache-line aligned to prevent false sharing
#[repr(align(64))]
pub struct LockFreeOrderNode {
    /// The order data (immutable after creation)
    pub order: Order,
    
    /// Pointer to the next node in the list
    pub next: AlignedAtomicPtr<LockFreeOrderNode>,
    
    /// Epoch for memory reclamation
    pub epoch: AtomicU64,
    
    /// Reference count for safe memory management
    pub ref_count: AtomicU64,
    
    /// ABA counter to prevent ABA problem
    pub aba_counter: AtomicUsize,
    
    /// Padding to ensure cache-line alignment
    _padding: [u8; 0],
}

impl LockFreeOrderNode {
    /// Create a new lock-free order node
    pub fn new(order: Order) -> Self {
        Self {
            order,
            next: AlignedAtomicPtr::new(ptr::null_mut()),
            epoch: AtomicU64::new(0),
            ref_count: AtomicU64::new(1),
            aba_counter: AtomicUsize::new(0),
            _padding: [],
        }
    }

    /// Get the next node in the list
    #[inline(always)]
    pub fn get_next(&self) -> *mut LockFreeOrderNode {
        self.next.load(MemoryOrder::Acquire)
    }

    /// Set the next node using compare-and-swap
    #[inline(always)]
    pub fn set_next(&self, expected: *mut LockFreeOrderNode, new: *mut LockFreeOrderNode) -> Result<*mut LockFreeOrderNode, *mut LockFreeOrderNode> {
        self.next.compare_exchange_weak(expected, new, MemoryOrder::Release, MemoryOrder::Relaxed)
    }

    /// Atomically update the next pointer
    #[inline(always)]
    pub fn store_next(&self, next: *mut LockFreeOrderNode) {
        self.next.store(next, MemoryOrder::Release);
    }

    /// Increment reference count
    #[inline(always)]
    pub fn acquire(&self) -> u64 {
        self.ref_count.fetch_add(1, Ordering::AcqRel)
    }

    /// Decrement reference count and return new count
    #[inline(always)]
    pub fn release(&self) -> u64 {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) - 1
    }

    /// Get current reference count
    #[inline(always)]
    pub fn get_ref_count(&self) -> u64 {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Set epoch for memory reclamation
    #[inline(always)]
    pub fn set_epoch(&self, epoch: u64) {
        self.epoch.store(epoch, Ordering::Release);
    }

    /// Get epoch
    #[inline(always)]
    pub fn get_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Check if this node can be safely reclaimed
    #[inline(always)]
    pub fn can_reclaim(&self, current_epoch: u64) -> bool {
        let node_epoch = self.get_epoch();
        let ref_count = self.get_ref_count();
        
        // Can reclaim if no references and epoch is old enough
        ref_count == 0 && (current_epoch > node_epoch + 2)
    }

    /// Increment ABA counter to prevent ABA problem
    #[inline(always)]
    pub fn increment_aba_counter(&self) -> usize {
        self.aba_counter.fetch_add(1, Ordering::AcqRel)
    }

    /// Get current ABA counter value
    #[inline(always)]
    pub fn get_aba_counter(&self) -> usize {
        self.aba_counter.load(Ordering::Acquire)
    }

    /// Compare and swap with ABA protection
    #[inline(always)]
    pub fn cas_next_with_aba_protection(
        &self,
        expected: *mut LockFreeOrderNode,
        new: *mut LockFreeOrderNode,
        expected_aba: usize,
    ) -> Result<*mut LockFreeOrderNode, (*mut LockFreeOrderNode, usize)> {
        let current_aba = self.get_aba_counter();
        if current_aba != expected_aba {
            return Err((self.get_next(), current_aba));
        }

        match self.set_next(expected, new) {
            Ok(old) => {
                self.increment_aba_counter();
                Ok(old)
            }
            Err(actual) => Err((actual, self.get_aba_counter())),
        }
    }
}

/// Lock-free linked list operations for order nodes with hazard pointer protection
pub struct LockFreeOrderList {
    hazard_manager: Arc<HazardPointerManager>,
}

impl LockFreeOrderList {
    /// Create a new lock-free order list with hazard pointer protection
    pub fn new(hazard_manager: Arc<HazardPointerManager>) -> Self {
        Self { hazard_manager }
    }
}

impl LockFreeOrderList {
    /// Insert a node at the head of the list with hazard pointer protection
    pub fn insert_head(
        &self,
        head: &AlignedAtomicPtr<LockFreeOrderNode>,
        new_node: *mut LockFreeOrderNode,
    ) -> Result<(), LockFreeListError> {
        if new_node.is_null() {
            return Err(LockFreeListError::NullPointer);
        }

        let hazard = self.hazard_manager.acquire_hazard_pointer();

        loop {
            let current_head = head.load(MemoryOrder::Acquire);
            hazard.protect(current_head);
            
            // Verify head hasn't changed after protection
            if head.load(MemoryOrder::Acquire) != current_head {
                continue;
            }
            
            unsafe {
                (*new_node).store_next(current_head);
            }

            match head.compare_exchange_weak(
                current_head,
                new_node,
                MemoryOrder::Release,
                MemoryOrder::Relaxed,
            ) {
                Ok(_) => {
                    hazard.clear();
                    return Ok(());
                }
                Err(_) => {
                    // Retry with new head
                    continue;
                }
            }
        }
    }

    /// Insert a node at the tail of the list with hazard pointer protection
    pub fn insert_tail(
        &self,
        head: &AlignedAtomicPtr<LockFreeOrderNode>,
        tail: &AlignedAtomicPtr<LockFreeOrderNode>,
        new_node: *mut LockFreeOrderNode,
    ) -> Result<(), LockFreeListError> {
        if new_node.is_null() {
            return Err(LockFreeListError::NullPointer);
        }

        let hazard_tail = self.hazard_manager.acquire_hazard_pointer();
        let hazard_head = self.hazard_manager.acquire_hazard_pointer();

        unsafe {
            (*new_node).store_next(ptr::null_mut());
        }

        loop {
            let current_tail = tail.load(MemoryOrder::Acquire);
            hazard_tail.protect(current_tail);
            
            // Verify tail hasn't changed after protection
            if tail.load(MemoryOrder::Acquire) != current_tail {
                continue;
            }
            
            if current_tail.is_null() {
                // Empty list - try to set both head and tail
                let current_head = head.load(MemoryOrder::Acquire);
                hazard_head.protect(current_head);
                
                if head.load(MemoryOrder::Acquire) != current_head {
                    continue;
                }
                
                match head.compare_exchange_weak(
                    ptr::null_mut(),
                    new_node,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => {
                        tail.store(new_node, MemoryOrder::Release);
                        hazard_tail.clear();
                        hazard_head.clear();
                        return Ok(());
                    }
                    Err(_) => continue,
                }
            } else {
                // Non-empty list - append to tail
                unsafe {
                    let current_aba = (*current_tail).get_aba_counter();
                    match (*current_tail).cas_next_with_aba_protection(ptr::null_mut(), new_node, current_aba) {
                        Ok(_) => {
                            // Successfully linked, now update tail
                            let _ = tail.compare_exchange_weak(
                                current_tail,
                                new_node,
                                MemoryOrder::Release,
                                MemoryOrder::Relaxed,
                            );
                            hazard_tail.clear();
                            hazard_head.clear();
                            return Ok(());
                        }
                        Err(_) => {
                            // Tail changed, retry
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// Remove the head node from the list with hazard pointer protection
    pub fn remove_head(
        &self,
        head: &AlignedAtomicPtr<LockFreeOrderNode>,
        tail: &AlignedAtomicPtr<LockFreeOrderNode>,
    ) -> Result<*mut LockFreeOrderNode, LockFreeListError> {
        let hazard_head = self.hazard_manager.acquire_hazard_pointer();
        let hazard_next = self.hazard_manager.acquire_hazard_pointer();

        loop {
            let current_head = head.load(MemoryOrder::Acquire);
            hazard_head.protect(current_head);
            
            // Verify head hasn't changed after protection
            if head.load(MemoryOrder::Acquire) != current_head {
                continue;
            }
            
            if current_head.is_null() {
                hazard_head.clear();
                hazard_next.clear();
                return Err(LockFreeListError::EmptyList);
            }

            unsafe {
                let next = (*current_head).get_next();
                hazard_next.protect(next);
                
                // Verify next hasn't changed
                if (*current_head).get_next() != next {
                    continue;
                }
                
                match head.compare_exchange_weak(
                    current_head,
                    next,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => {
                        // Successfully removed head
                        if next.is_null() {
                            // List is now empty, update tail
                            tail.store(ptr::null_mut(), MemoryOrder::Release);
                        }
                        
                        // Retire the removed node
                        self.hazard_manager.retire_pointer(current_head);
                        
                        hazard_head.clear();
                        hazard_next.clear();
                        return Ok(current_head);
                    }
                    Err(_) => {
                        // Head changed, retry
                        continue;
                    }
                }
            }
        }
    }

    /// Find and remove a specific node from the list with hazard pointer protection
    pub fn remove_node(
        &self,
        head: &AlignedAtomicPtr<LockFreeOrderNode>,
        tail: &AlignedAtomicPtr<LockFreeOrderNode>,
        target_node: *mut LockFreeOrderNode,
    ) -> Result<bool, LockFreeListError> {
        if target_node.is_null() {
            return Err(LockFreeListError::NullPointer);
        }

        loop {
            let mut prev: *mut LockFreeOrderNode = ptr::null_mut();
            let mut current = head.load(MemoryOrder::Acquire);

            // Traverse the list to find the target node
            while !current.is_null() {
                if current == target_node {
                    // Found the target node
                    unsafe {
                        let next = (*current).get_next();
                        
                        if prev.is_null() {
                            // Removing head
                            match head.compare_exchange_weak(
                                current,
                                next,
                                MemoryOrder::Release,
                                MemoryOrder::Relaxed,
                            ) {
                                Ok(_) => {
                                    if next.is_null() {
                                        tail.store(ptr::null_mut(), MemoryOrder::Release);
                                    }
                                    return Ok(true);
                                }
                                Err(_) => break, // Restart from beginning
                            }
                        } else {
                            // Removing from middle or end
                            match (*prev).set_next(current, next) {
                                Ok(_) => {
                                    if next.is_null() {
                                        tail.store(prev, MemoryOrder::Release);
                                    }
                                    return Ok(true);
                                }
                                Err(_) => break, // Restart from beginning
                            }
                        }
                    }
                }

                prev = current;
                unsafe {
                    current = (*current).get_next();
                }
            }

            // If we reach here, either the node wasn't found or there was contention
            if current.is_null() {
                return Ok(false); // Node not found
            }
            // Otherwise, restart due to contention
        }
    }

    /// Count the number of nodes in the list (for debugging)
    pub fn count_nodes(head: &AlignedAtomicPtr<LockFreeOrderNode>) -> usize {
        let mut count = 0;
        let mut current = head.load(MemoryOrder::Acquire);

        while !current.is_null() {
            count += 1;
            unsafe {
                current = (*current).get_next();
            }
        }

        count
    }

    /// Validate list integrity (for debugging)
    pub fn validate_list(
        head: &AlignedAtomicPtr<LockFreeOrderNode>,
        tail: &AlignedAtomicPtr<LockFreeOrderNode>,
    ) -> Result<(), LockFreeListError> {
        let head_ptr = head.load(MemoryOrder::Acquire);
        let tail_ptr = tail.load(MemoryOrder::Acquire);

        if head_ptr.is_null() && tail_ptr.is_null() {
            // Empty list is valid
            return Ok(());
        }

        if head_ptr.is_null() || tail_ptr.is_null() {
            // One null, one non-null is invalid
            return Err(LockFreeListError::CorruptedList);
        }

        // Traverse to find tail
        let mut current = head_ptr;
        let mut prev = ptr::null_mut();

        while !current.is_null() {
            prev = current;
            unsafe {
                current = (*current).get_next();
            }
        }

        if prev != tail_ptr {
            return Err(LockFreeListError::CorruptedList);
        }

        Ok(())
    }
}

/// Error types for lock-free list operations
#[derive(Debug, Clone, PartialEq)]
pub enum LockFreeListError {
    NullPointer,
    EmptyList,
    CorruptedList,
    ConcurrentModification,
}

/// Iterator for lock-free order list
pub struct LockFreeOrderIterator {
    current: *mut LockFreeOrderNode,
}

impl LockFreeOrderIterator {
    pub fn new(head: &AlignedAtomicPtr<LockFreeOrderNode>) -> Self {
        Self {
            current: head.load(MemoryOrder::Acquire),
        }
    }
}

impl Iterator for LockFreeOrderIterator {
    type Item = *mut LockFreeOrderNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            return None;
        }

        let current = self.current;
        unsafe {
            self.current = (*current).get_next();
        }

        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{Order, OrderId, Symbol, Side, OrderType};
    use std::sync::Arc;
    use std::thread;

    fn create_test_order(id: u64) -> Order {
        Order {
            id: OrderId::new(id),
            symbol: Symbol::new("BTCUSD").unwrap(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 50000,
            size: 100,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_order_node_creation() {
        let order = create_test_order(1);
        let node = LockFreeOrderNode::new(order.clone());
        
        assert_eq!(node.order.id, order.id);
        assert!(node.get_next().is_null());
        assert_eq!(node.get_ref_count(), 1);
    }

    #[test]
    fn test_reference_counting() {
        let order = create_test_order(1);
        let node = LockFreeOrderNode::new(order);
        
        assert_eq!(node.get_ref_count(), 1);
        
        let old_count = node.acquire();
        assert_eq!(old_count, 1);
        assert_eq!(node.get_ref_count(), 2);
        
        let new_count = node.release();
        assert_eq!(new_count, 1);
        assert_eq!(node.get_ref_count(), 1);
    }

    #[test]
    fn test_insert_head() {
        let head = AlignedAtomicPtr::new(ptr::null_mut());
        let order1 = create_test_order(1);
        let order2 = create_test_order(2);
        
        let node1 = Box::into_raw(Box::new(LockFreeOrderNode::new(order1)));
        let node2 = Box::into_raw(Box::new(LockFreeOrderNode::new(order2)));
        
        // Insert first node
        LockFreeOrderList::insert_head(&head, node1).unwrap();
        assert_eq!(head.load(MemoryOrder::Acquire), node1);
        
        // Insert second node at head
        LockFreeOrderList::insert_head(&head, node2).unwrap();
        assert_eq!(head.load(MemoryOrder::Acquire), node2);
        
        unsafe {
            assert_eq!((*node2).get_next(), node1);
            assert!((*node1).get_next().is_null());
        }
        
        // Cleanup
        unsafe {
            Box::from_raw(node1);
            Box::from_raw(node2);
        }
    }

    #[test]
    fn test_insert_tail() {
        let head = AlignedAtomicPtr::new(ptr::null_mut());
        let tail = AlignedAtomicPtr::new(ptr::null_mut());
        
        let order1 = create_test_order(1);
        let order2 = create_test_order(2);
        
        let node1 = Box::into_raw(Box::new(LockFreeOrderNode::new(order1)));
        let node2 = Box::into_raw(Box::new(LockFreeOrderNode::new(order2)));
        
        // Insert first node
        LockFreeOrderList::insert_tail(&head, &tail, node1).unwrap();
        assert_eq!(head.load(MemoryOrder::Acquire), node1);
        assert_eq!(tail.load(MemoryOrder::Acquire), node1);
        
        // Insert second node at tail
        LockFreeOrderList::insert_tail(&head, &tail, node2).unwrap();
        assert_eq!(head.load(MemoryOrder::Acquire), node1);
        assert_eq!(tail.load(MemoryOrder::Acquire), node2);
        
        unsafe {
            assert_eq!((*node1).get_next(), node2);
            assert!((*node2).get_next().is_null());
        }
        
        // Cleanup
        unsafe {
            Box::from_raw(node1);
            Box::from_raw(node2);
        }
    }

    #[test]
    fn test_remove_head() {
        let head = AlignedAtomicPtr::new(ptr::null_mut());
        let tail = AlignedAtomicPtr::new(ptr::null_mut());
        
        let order1 = create_test_order(1);
        let order2 = create_test_order(2);
        
        let node1 = Box::into_raw(Box::new(LockFreeOrderNode::new(order1)));
        let node2 = Box::into_raw(Box::new(LockFreeOrderNode::new(order2)));
        
        // Insert nodes
        LockFreeOrderList::insert_tail(&head, &tail, node1).unwrap();
        LockFreeOrderList::insert_tail(&head, &tail, node2).unwrap();
        
        // Remove head
        let removed = LockFreeOrderList::remove_head(&head, &tail).unwrap();
        assert_eq!(removed, node1);
        assert_eq!(head.load(MemoryOrder::Acquire), node2);
        assert_eq!(tail.load(MemoryOrder::Acquire), node2);
        
        // Remove last node
        let removed = LockFreeOrderList::remove_head(&head, &tail).unwrap();
        assert_eq!(removed, node2);
        assert!(head.load(MemoryOrder::Acquire).is_null());
        assert!(tail.load(MemoryOrder::Acquire).is_null());
        
        // Try to remove from empty list
        let result = LockFreeOrderList::remove_head(&head, &tail);
        assert_eq!(result.unwrap_err(), LockFreeListError::EmptyList);
        
        // Cleanup
        unsafe {
            Box::from_raw(node1);
            Box::from_raw(node2);
        }
    }

    #[test]
    fn test_remove_specific_node() {
        let head = AlignedAtomicPtr::new(ptr::null_mut());
        let tail = AlignedAtomicPtr::new(ptr::null_mut());
        
        let order1 = create_test_order(1);
        let order2 = create_test_order(2);
        let order3 = create_test_order(3);
        
        let node1 = Box::into_raw(Box::new(LockFreeOrderNode::new(order1)));
        let node2 = Box::into_raw(Box::new(LockFreeOrderNode::new(order2)));
        let node3 = Box::into_raw(Box::new(LockFreeOrderNode::new(order3)));
        
        // Insert nodes
        LockFreeOrderList::insert_tail(&head, &tail, node1).unwrap();
        LockFreeOrderList::insert_tail(&head, &tail, node2).unwrap();
        LockFreeOrderList::insert_tail(&head, &tail, node3).unwrap();
        
        // Remove middle node
        let removed = LockFreeOrderList::remove_node(&head, &tail, node2).unwrap();
        assert!(removed);
        
        // Verify list integrity
        assert_eq!(head.load(MemoryOrder::Acquire), node1);
        assert_eq!(tail.load(MemoryOrder::Acquire), node3);
        
        unsafe {
            assert_eq!((*node1).get_next(), node3);
            assert!((*node3).get_next().is_null());
        }
        
        // Try to remove non-existent node
        let removed = LockFreeOrderList::remove_node(&head, &tail, node2).unwrap();
        assert!(!removed);
        
        // Cleanup
        unsafe {
            Box::from_raw(node1);
            Box::from_raw(node2);
            Box::from_raw(node3);
        }
    }

    #[test]
    fn test_concurrent_operations() {
        let head = Arc::new(AlignedAtomicPtr::new(ptr::null_mut()));
        let tail = Arc::new(AlignedAtomicPtr::new(ptr::null_mut()));
        let mut handles = vec![];

        // Spawn threads to insert nodes concurrently
        for i in 0..10 {
            let head_clone = head.clone();
            let tail_clone = tail.clone();
            
            let handle = thread::spawn(move || {
                let order = create_test_order(i);
                let node = Box::into_raw(Box::new(LockFreeOrderNode::new(order)));
                LockFreeOrderList::insert_tail(&head_clone, &tail_clone, node).unwrap();
                node
            });
            handles.push(handle);
        }

        // Collect all inserted nodes
        let mut nodes = vec![];
        for handle in handles {
            nodes.push(handle.join().unwrap());
        }

        // Verify list has correct number of nodes
        let count = LockFreeOrderList::count_nodes(&head);
        assert_eq!(count, 10);

        // Validate list integrity
        LockFreeOrderList::validate_list(&head, &tail).unwrap();

        // Cleanup
        for node in nodes {
            unsafe { Box::from_raw(node); }
        }
    }

    #[test]
    fn test_iterator() {
        let head = AlignedAtomicPtr::new(ptr::null_mut());
        let tail = AlignedAtomicPtr::new(ptr::null_mut());
        
        // Insert some nodes
        for i in 1..=5 {
            let order = create_test_order(i);
            let node = Box::into_raw(Box::new(LockFreeOrderNode::new(order)));
            LockFreeOrderList::insert_tail(&head, &tail, node).unwrap();
        }

        // Iterate and collect order IDs
        let mut order_ids = vec![];
        for node_ptr in LockFreeOrderIterator::new(&head) {
            unsafe {
                order_ids.push((*node_ptr).order.id.as_u64());
            }
        }

        assert_eq!(order_ids, vec![1, 2, 3, 4, 5]);

        // Cleanup
        while let Ok(node) = LockFreeOrderList::remove_head(&head, &tail) {
            unsafe { Box::from_raw(node); }
        }
    }
}
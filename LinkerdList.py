def copyRandomList(head: 'Node'):
    cur = head
    copyHead = Node(-1)
    copyCur = copyHead
    while cur:
        copyCur.next = Node(cur.val)
        nextCur = cur.next
        cur.next = copyCur.next
        copyCur.next.next = nextCur
        cur = nextCur

    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
            cur = cur.next.next

    cur = head
    copyCur = copyHead
    while cur:
        copyCur.next = cur.next
        cur = cur.next.next
        copyCur = copyCur.next

    return copyHead.next
def unicode_airlock(v):
    if isinstance(v, bytes):
        v = v.decode()
    return v

def ascii_airlock(v):
    if not isinstance(v, bytes):
        v = v.encode()
    return v

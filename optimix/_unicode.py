def unicode_airlock(v):
    if isinstance(v, bytes):
        v = v.decode()
    return v

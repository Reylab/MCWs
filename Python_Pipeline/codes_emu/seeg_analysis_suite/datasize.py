one_day_128_micros_bytes = 30000 * 2 * 24 * 3600 * 128
one_day_128_micros_GB = one_day_128_micros_bytes / 1024**3
one_day_128_micros_TB = one_day_128_micros_GB / 1024

print(f'24 hrs recording with 128 micros: {one_day_128_micros_GB} GB')

one_day_256_macros_bytes = 2000 * 4 * 3600 * 256
one_day_256_macros_GB = one_day_256_macros_bytes / 1024**3
one_day_256_macros_TB = one_day_256_macros_GB / 1024

print(f'24 hrs recording with 256 macros: {one_day_256_macros_GB} GB')

print(f'24hrs Total: {one_day_128_micros_GB + one_day_256_macros_GB} GB')
print(f'One week: {(one_day_128_micros_TB + one_day_256_macros_TB) * 7} TB')

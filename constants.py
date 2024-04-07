def color_8bit():
    return 255
def byte_to_bit():
    return 8
def hex_to_dec():
    return 16
def color_12bit():
    return 4095
def version():
    return '01'
def header_lead():
    return '42'
def header_name():
    return '5044696779'
def header():
    return {0:[ header_lead(),1], 1:[ version(),1], 2:[ header_name(),5]}
def mum_to_inch():
    return 25400
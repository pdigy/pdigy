def color_8bit():
    return 255
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

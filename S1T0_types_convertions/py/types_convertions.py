import ctypes


def dec2bin(x, signed) -> str:
    if x == 0:
        return '0' * 32
    answer = ''
    y = abs(x)
    while y != 1:
        answer += str(y % 2)
        y //= 2
    answer = ('0' * (31 - len(answer))) + '1' + answer[::-1]
    if signed:
        if x < 0:
            opposite = ''
            last_zero = 0
            for i in range(len(answer)):
                if answer[i] == '1':
                    opposite += '0'
                    last_zero = i
                else:
                    opposite += '1'
            answer = opposite[:last_zero] + '1' + '0' * (len(answer) - last_zero - 1)
        return answer
    else:
        return '-' * (x < 0) + '0' * ((x > 0) & (x < 2 ** 31)) + '1' * (x >= 2 ** 31) + answer[1:]


def bin2dec(x, signed) -> int:
    answer = 0
    for i in x:
        answer = answer * 2 + int(i)
    return answer - 2 ** 32 * int(x[0]) * signed


def float2bin(x) -> str:
    """
    get binary representation of float32
    you may leave it as is
    """
    return f'{ctypes.c_uint32.from_buffer(ctypes.c_float(x)).value:>032b}'


def bin2float(x) -> float:
    answer = 0
    if x[1:].count('1') == 0:
        return 0
    y = '1' + x[9:]
    k = bin2dec(x[1:9], False) - 127
    for i in y:
        answer += int(i) * (2 ** k)
        k -= 1
    if x[0] == '1':
        answer *= (-1)
    return answer

import os
from os.path import exists
def main():
    i   = 2

    # for i in range(10):
    while i< 6:
        print('trying',i)
        try:
            cmd = "python sac.py"
            os.system(cmd)

            if exists('TEMP'):
                print('saving.............', i)
                name='./records/SAC_'+str(i)+'.csv'
                os.rename('TEMP',name)
                i+=1
        except:
            pass


if __name__ == '__main__':
    main()
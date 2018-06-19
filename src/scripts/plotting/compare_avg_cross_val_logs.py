import sys
from we_panic_utils.basic_utils.graphing import compare_losses

def usage(with_help=True): 
    print("[Usage]: %s <input_model_dir_1> ... <input_model_dir_n> <output_fig_dir>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()


def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0]))

    print("\t| script meant to take multiple run_history directories")
    print("\t| and plot the average loss among multiple models")
    usage(with_help=False)

if __name__ == '__main__':
    try:
        if 'help' == sys.argv[1] or 'HELP' == sys.argv[1] or 'h' == sys.argv[1]:
            help_msg()

        dirs_in  =   sys.argv[1:-1]
        dir_out  =   sys.argv[-1]
        if len(dirs_in) < 2:
            raise Exception

    except Exception as e:
        usage()

    compare_losses(dirs_in, dir_out)

import sys
from we_panic_utils.basic_utils.graphing import plot_multiple_losses

def usage(with_help=True): 
    print("[Usage]: %s <input_model_dir> <output_fig_dir>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()


def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0]))

    print("\t| script meant to take the run_history directory of a")
    print("\t| model that has been cross validated and plots each fold")
    usage(with_help=False)

if __name__ == '__main__':
    try:
        if 'help' == sys.argv[1] or 'HELP' == sys.argv[1] or 'h' == sys.argv[1]:
            help_msg()

        dir_in  =   sys.argv[1]
        dir_out = None
        if len(sys.argv) > 2:
            dir_out =   sys.argv[2]
    except Exception as e:
        usage()

    plot_multiple_losses(dir_in, dir_out)

from __future__ import absolute_import

try:
    from progressbar import ProgressBar as _ProgressBar
    from progressbar import Bar, ETA, Percentage, ReverseBar, signal

    class ProgressBar(_ProgressBar):
        "overriding the default to delete the progress bar when finished"
        def finish(self):
            'Puts the ProgressBar bar in the finished state.'
            self.finished = True
            self.update(self.maxval)
            # clear the progress bar:
            self.fd.write('\r'+' '*self.term_width+'\r')
            if self.signal_set:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)

except:
    # Dummy no-op progress bar to simplify code using ProgressBar
    class ProgressBar(object):
        def __init__(*args, **kwargs):
            pass

        def start(self):
            return self

        def finish(self):
            pass

        def update(self, x):
            pass

    Bar = ETA = Percentage = ReverseBar = ProgressBar

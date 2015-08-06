import sublime, sublime_plugin
import subprocess, os

class SphinxPreviewCommand(sublime_plugin.TextCommand):
    def run(self, edit, **kwargs):
        if self.view.file_name():
            folder_name, file_name = os.path.split(self.view.file_name())

        command = './build_and_view.sh'
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder_name, shell=True)
        result, err = p.communicate()
        print(result,err)
        # self.view.set_status('p4',str(result+err))
        # sublime.set_timeout(self.clear,2000)
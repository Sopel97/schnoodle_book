import glob

files = glob.glob('./Schnoodle Book*')

files.sort(key=lambda x: int(x.split(' ')[4]))

with open('index.html', 'w') as f:
    f.write('''<!doctype html>
<html>
  <head>
    <title>This is the title of the webpage!</title>
  </head>
  <body>
''')
    f.write(f'<p><a href="https://github.com/Sopel97/schnoodle_book">Github repo (including README)</a></p>')
    f.write(f'<p>Recommended to use firefox or chrome. Rendering may be wrong on other browsers.</p>')

    for file in files:
        f.write(f'<a href="{file}">')
        f.write(file[2:-5])
        f.write('</a></br>')
    f.write('''</body>
</html>''')

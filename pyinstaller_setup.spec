# pyinstaller_setup.spec
# Run: pyinstaller pyinstaller_setup.spec
a = Analysis(['swmmanywhere/__main__.py'],
             pathex=['.'],
             binaries=[],
             datas=[],
             hiddenimports=['shapely', 'geopandas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=None)
             
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='swmmanywhere',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )

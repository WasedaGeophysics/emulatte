1. pytestの実行用ディレクトリ
   ================

- [1. pytestの実行用ディレクトリ](#1-pytestの実行用ディレクトリ)
- [2. 解析解によるテスト](#2-解析解によるテスト)
  - [2.1. 垂直磁気双極子 (Vertical Magnetic Dipole: VMD)](#21-垂直磁気双極子-vertical-magnetic-dipole-vmd)
  - [2.2. 円形ループ (Circular Loop: CL)](#22-円形ループ-circular-loop-cl)

2. 解析解によるテスト
   =========

2.1. 垂直磁気双極子 (Vertical Magnetic Dipole: VMD)
--------------------------------------------

- 周波数領域
  - 2021-12-14：skosaka
- 時間領域

  - 2021-12-14：skosaka

    - DLAGの修正
    - ignore_displacement_current=True
    - hankel_filterの違いによるpytest合格可否

2.2. 円形ループ (Circular Loop: CL)
------------------------------

- 周波数領域
  - 2021-12-14：skosaka
- 時間領域
  - 2021-12-14：skosaka
    - 応答自体が小さいためか、フィッティングはいいが相対誤差が大きくなる。tol_errの設定に悩む

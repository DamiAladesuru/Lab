@echo off
set ds=%~1
set gaul=%~2
set eez=%~3
set res=%~4

pgsql2shp -f %ds%_%res%km.shp -h iguana.eea.europa.eu -u michimau gis_sdi "select distinct x.cellcode, x.eoforigin, x.noforigin, x.the_geom from ( select distinct g1.cellcode, g1.eoforigin, g1.noforigin, g1.the_geom from grids.inspire_grid_%res%km g1, (select distinct a.adm0_name, st_buffer(st_transform(a.wkb_geometry,3035),15000) as wkb_geometry from gaul.g2009_2008_0 a where a.adm0_name = '%gaul%') b where st_intersects(g1.the_geom, b.wkb_geometry) UNION select g2.cellcode, g2.eoforigin, g2.noforigin, g2.the_geom from grids.inspire_grid_%res%km g2, (select distinct d.country, st_buffer(st_transform(d.geom,3035),15000) as geom from eez.world_eez_v7_2012_hr d where d.country = '%eez%') e where st_intersects(g2.the_geom, e.geom)) x"

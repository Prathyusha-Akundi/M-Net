function volume2 = map_intensities(volume,actual,map)
    volume2 = zeros(size(volume));
    for i=1:length(actual)
       iseq = volume==actual(i);
       volume2(iseq) = map(i);
    end
end
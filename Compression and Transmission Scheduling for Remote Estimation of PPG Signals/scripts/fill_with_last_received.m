function out = fill_with_last_received(pairs, amplitude_column, simulation_time)
  % Given a set of [value, time] packets that were successfully received
  % (possibly out of order and possibly only covering part of the
  % timeline), reconstruct a full-length series by holding the last
  % received value at every time step, with the median of the true
  % signal as the fallback before the first packet arrives.
  out = nan(1, simulation_time);
  if isempty(pairs)
    out(:) = median(amplitude_column);
    return;
  end
  pairs = sortrows(pairs, 2);
  last_value = median(amplitude_column);
  pi_idx = 1;
  for i = 1:simulation_time
    while pi_idx <= size(pairs,1) && pairs(pi_idx,2) <= i
      last_value = pairs(pi_idx,1);
      pi_idx = pi_idx + 1;
    end
    out(i) = last_value;
  end
end

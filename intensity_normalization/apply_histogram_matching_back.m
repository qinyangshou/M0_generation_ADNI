function [img_out] = apply_histogram_matching_back(img_in, landmarks_ori, landmarks_norm, methodT)

    T = landmarks_ori;
    m_k = landmarks_norm;
    [~, ind] = unique( T.values);
    if strcmp(methodT,'linear')
	    
	    img_out = interp1([0, m_k.landmarks(ind), m_k.maxI],[0, T.values(ind), T.maxI], img_in) ;
    elseif strcmp(methodT,'spline')
        %img_out = spline( [minT T.values maxT], [SscaleExtremeMin m_k.landmarks SscaleExtremeMax], img_in);
        img_out = spline([0, m_k.landmarks(ind), m_k.maxI],[0, T.values(ind), T.maxI], img_in);
    else
        error('Invalid value for Method')
    end       
    img_out(isnan(img_out))=0;
end
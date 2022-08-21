def skeleton():
    skeleton = face()+pose()+r_hand()+l_hand()
    V_skel =set()
    for i in skeleton:
        V_skel.update(i)
    assert len(V_skel) == 137
    return skeleton

def face():
    face_outline = [(i,i+1) for i in range(0,16)]
    eye_brow = [(i,i+1) for i in range(17,26)]
    eye_l = [(i,i+1) for i in range(36,41)] + [(36,41)] + [(68,69),]
    eye_r = [(i,i+1) for i in range(42,47)] + [(42,47)]
    nose = [(i,i+1) for i in range(27,30)]
    nose_under = [(i,i+1) for i in range(31,35)]
    mouse = [(i,i+1) for i in range(48,59)] + [(48,59)]+ [(i,i+1) for i in range(60,67)] + [(60,67)]

    face = face_outline + eye_brow + eye_l+ eye_r+ nose + nose_under + mouse
    V_face = set()
    for i in face:
        V_face.update(i)
    assert len(V_face) == 70
    
    return face

def pose():
    pose = [(70,71), (71,78),] + [(85,87),(85,70),(70,86),(86,88)] #face and spine
    pose = pose + [(i,i+1) for i in range(71,74)] #left arm
    pose = pose + [(71,75),] + [(i,i+1) for i in range(75,77)] #right arm
    pose = pose + [(78,79),(78,82),(79,80),(80,81),(82,83),(83,84)] #legs
    pose = pose + [(92,93),(92,94),(89,90),(89,91)] #foot?
    V_pose =set()
    for i in pose:
        V_pose.update(i)
    assert len(V_pose) == 25
    return pose

def l_hand():
    offset=95
    l_hand = [(offset,offset+i) for i in range(1,20,4)]
    l_hand = l_hand + [(offset+i,offset+i+1) for i in range(1,4)]
    l_hand = l_hand + [(offset+i,offset+i+1) for i in range(5,8)]
    l_hand = l_hand + [(offset+i,offset+i+1) for i in range(9,12)]
    l_hand = l_hand + [(offset+i,offset+i+1) for i in range(13,16)]
    l_hand = l_hand + [(offset+i,offset+i+1) for i in range(17,20)]
    V_l_hand =set()
    for i in l_hand:
        V_l_hand.update(i)
    assert len(V_l_hand) == 21
    return l_hand

def r_hand():
    offset=116
    r_hand = [(offset,offset+i) for i in range(1,20,4)]
    r_hand = r_hand + [(offset+i,offset+i+1) for i in range(1,4)]
    r_hand = r_hand + [(offset+i,offset+i+1) for i in range(5,8)]
    r_hand = r_hand + [(offset+i,offset+i+1) for i in range(9,12)]
    r_hand = r_hand + [(offset+i,offset+i+1) for i in range(13,16)]
    r_hand = r_hand + [(offset+i,offset+i+1) for i in range(17,20)]
    V_r_hand =set()
    for i in r_hand:
        V_r_hand.update(i)
    assert len(V_r_hand) == 21
    return r_hand

if __name__ == '__main__':
    skeleton()
    pass
import pickle
import gzip

for path in [42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54]:
	save_path = 'out/basic/19/eval/test-0%s000.pklz' % path#'out/basic/06/eval/dev-040000.pklz'#'out/basic/12/eval/dev-047000.pklz'#out/basic/10/eval/dev-053000.pklz'#'out/basic/09/eval/dev-042000.pklz' #'out/basic/06/eval/dev-040000.pklz'
	f = gzip.open(save_path,'rb')
	res= pickle.load(f)
	f.close()

	print(save_path)
	print(res['f1'])
	print(res['acc'])


#restore the object
#out/basic/19/eval
for path in ['041000']:
	save_path = 'out/basic/17/eval/test-%s.pklz' % path#'out/basic/06/eval/dev-040000.pklz'#'out/basic/12/eval/dev-047000.pklz'#out/basic/10/eval/dev-053000.pklz'#'out/basic/09/eval/dev-042000.pklz' #'out/basic/06/eval/dev-040000.pklz'
	f = gzip.open(save_path,'rb')
	res= pickle.load(f)
	f.close()

	print(save_path)
	print(res['f1'])
	print(res['acc'])

for path in ['041000', '042000', '043000', '044000', '045000']:
	save_path = 'out/basic/14/eval/test-%s.pklz' % path#'out/basic/06/eval/dev-040000.pklz'#'out/basic/12/eval/dev-047000.pklz'#out/basic/10/eval/dev-053000.pklz'#'out/basic/09/eval/dev-042000.pklz' #'out/basic/06/eval/dev-040000.pklz'
	f = gzip.open(save_path,'rb')
	res= pickle.load(f)
	f.close()

	print(save_path)
	print(res['f1'])
	print(res['acc'])

# out/basic/25/eval
for path in ['044000', '045000', '046000', '047000', '048000', '049000', '050000', '051000', '052000']:
	save_path = 'out/basic/14/eval/dev-%s.pklz' % path#'out/basic/06/eval/dev-040000.pklz'#'out/basic/12/eval/dev-047000.pklz'#out/basic/10/eval/dev-053000.pklz'#'out/basic/09/eval/dev-042000.pklz' #'out/basic/06/eval/dev-040000.pklz'
	f = gzip.open(save_path,'rb')
	res= pickle.load(f)
	f.close()

	print(save_path)
	print(res['f1'])
	print(res['acc'])

for path in ['041000', '042000']:
	save_path = 'out/basic/18/eval/dev-%s.pklz' % path#'out/basic/06/eval/dev-040000.pklz'#'out/basic/12/eval/dev-047000.pklz'#out/basic/10/eval/dev-053000.pklz'#'out/basic/09/eval/dev-042000.pklz' #'out/basic/06/eval/dev-040000.pklz'
	f = gzip.open(save_path,'rb')
	res= pickle.load(f)
	f.close()

	print(save_path)
	print(res['f1'])
	print(res['acc'])

for path in ['041000', '042000', '043000', '044000', '045000', '046000', '047000', '048000', '049000']:
	save_path = 'out/basic/17/eval/dev-%s.pklz' % path#'out/basic/06/eval/dev-040000.pklz'#'out/basic/12/eval/dev-047000.pklz'#out/basic/10/eval/dev-053000.pklz'#'out/basic/09/eval/dev-042000.pklz' #'out/basic/06/eval/dev-040000.pklz'
	f = gzip.open(save_path,'rb')
	res= pickle.load(f)
	f.close()

	print(save_path)
	print(res['f1'])
	print(res['acc'])
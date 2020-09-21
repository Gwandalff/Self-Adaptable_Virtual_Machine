package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.Assignment;
import miniJava.visitor.miniJava.Block;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.ForStatement;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

public class ForStatementImpl extends StatementImpl implements ForStatement {
	protected Assignment declaration;

	protected Expression condition;

	protected Assignment progression;

	protected Block block;

	protected ForStatementImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.FOR_STATEMENT;
	}

	public Assignment getDeclaration() {
		return declaration;
	}

	public NotificationChain basicSetDeclaration(Assignment newDeclaration, NotificationChain msgs) {
		Assignment oldDeclaration = declaration;
		declaration = newDeclaration;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__DECLARATION, oldDeclaration, newDeclaration);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setDeclaration(Assignment newDeclaration) {
		if (newDeclaration != declaration) {
			NotificationChain msgs = null;
			if (declaration != null)
				msgs = ((InternalEObject) declaration).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__DECLARATION, null, msgs);
			if (newDeclaration != null)
				msgs = ((InternalEObject) newDeclaration).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__DECLARATION, null, msgs);
			msgs = basicSetDeclaration(newDeclaration, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__DECLARATION, newDeclaration, newDeclaration));
	}

	public Expression getCondition() {
		return condition;
	}

	public NotificationChain basicSetCondition(Expression newCondition, NotificationChain msgs) {
		Expression oldCondition = condition;
		condition = newCondition;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__CONDITION, oldCondition, newCondition);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setCondition(Expression newCondition) {
		if (newCondition != condition) {
			NotificationChain msgs = null;
			if (condition != null)
				msgs = ((InternalEObject) condition).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__CONDITION, null, msgs);
			if (newCondition != null)
				msgs = ((InternalEObject) newCondition).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__CONDITION, null, msgs);
			msgs = basicSetCondition(newCondition, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__CONDITION, newCondition, newCondition));
	}

	public Assignment getProgression() {
		return progression;
	}

	public NotificationChain basicSetProgression(Assignment newProgression, NotificationChain msgs) {
		Assignment oldProgression = progression;
		progression = newProgression;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__PROGRESSION, oldProgression, newProgression);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setProgression(Assignment newProgression) {
		if (newProgression != progression) {
			NotificationChain msgs = null;
			if (progression != null)
				msgs = ((InternalEObject) progression).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__PROGRESSION, null, msgs);
			if (newProgression != null)
				msgs = ((InternalEObject) newProgression).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__PROGRESSION, null, msgs);
			msgs = basicSetProgression(newProgression, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__PROGRESSION, newProgression, newProgression));
	}

	public Block getBlock() {
		return block;
	}

	public NotificationChain basicSetBlock(Block newBlock, NotificationChain msgs) {
		Block oldBlock = block;
		block = newBlock;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__BLOCK, oldBlock, newBlock);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setBlock(Block newBlock) {
		if (newBlock != block) {
			NotificationChain msgs = null;
			if (block != null)
				msgs = ((InternalEObject) block).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__BLOCK, null, msgs);
			if (newBlock != null)
				msgs = ((InternalEObject) newBlock).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.FOR_STATEMENT__BLOCK, null, msgs);
			msgs = basicSetBlock(newBlock, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.FOR_STATEMENT__BLOCK, newBlock, newBlock));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.FOR_STATEMENT__DECLARATION :
				return basicSetDeclaration(null, msgs);
			case MiniJavaPackage.FOR_STATEMENT__CONDITION :
				return basicSetCondition(null, msgs);
			case MiniJavaPackage.FOR_STATEMENT__PROGRESSION :
				return basicSetProgression(null, msgs);
			case MiniJavaPackage.FOR_STATEMENT__BLOCK :
				return basicSetBlock(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.FOR_STATEMENT__DECLARATION :
				return getDeclaration();
			case MiniJavaPackage.FOR_STATEMENT__CONDITION :
				return getCondition();
			case MiniJavaPackage.FOR_STATEMENT__PROGRESSION :
				return getProgression();
			case MiniJavaPackage.FOR_STATEMENT__BLOCK :
				return getBlock();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.FOR_STATEMENT__DECLARATION :
				setDeclaration((Assignment) newValue);
				return;
			case MiniJavaPackage.FOR_STATEMENT__CONDITION :
				setCondition((Expression) newValue);
				return;
			case MiniJavaPackage.FOR_STATEMENT__PROGRESSION :
				setProgression((Assignment) newValue);
				return;
			case MiniJavaPackage.FOR_STATEMENT__BLOCK :
				setBlock((Block) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.FOR_STATEMENT__DECLARATION :
				setDeclaration((Assignment) null);
				return;
			case MiniJavaPackage.FOR_STATEMENT__CONDITION :
				setCondition((Expression) null);
				return;
			case MiniJavaPackage.FOR_STATEMENT__PROGRESSION :
				setProgression((Assignment) null);
				return;
			case MiniJavaPackage.FOR_STATEMENT__BLOCK :
				setBlock((Block) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.FOR_STATEMENT__DECLARATION :
				return declaration != null;
			case MiniJavaPackage.FOR_STATEMENT__CONDITION :
				return condition != null;
			case MiniJavaPackage.FOR_STATEMENT__PROGRESSION :
				return progression != null;
			case MiniJavaPackage.FOR_STATEMENT__BLOCK :
				return block != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__ForStatement(this);
	}
}
